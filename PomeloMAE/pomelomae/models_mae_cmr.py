from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class CMRMaskedAutoencoderViT(nn.Module):
    """MAE with center-prior complementary masking and multi-view pretraining.

    The encoder keeps the original MAE/ViT parameter names so its checkpoint can
    be loaded by the unchanged single-image finetuning code.

    During multi-view token-fusion pretraining, an intermediate encoder feature
    can be injected into the decoder through a deep-guided residual gate.  The
    added modules are pretraining-only and are not used by single-image fine-tuning.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 num_views=4, center_ratio=0.72, complementary_bias=1.5,
                 context_dropout=0.1, cross_layer_index=6,
                 cross_layer_dropout=0.2, cross_layer_gamma_init=0.1,
                 cross_layer_fusion_mode='gated',
                 subject_patch_threshold=0.25,
                 subject_center_mode='mean',
                 subject_visible_quadrant_ratio=0.70,
                 subject_visible_outside_mode='random'):
        super().__init__()

        self.num_views = num_views
        self.center_ratio = center_ratio
        self.complementary_bias = complementary_bias
        self.context_dropout = context_dropout
        if not 0.0 <= subject_patch_threshold <= 1.0:
            raise ValueError(
                "subject_patch_threshold must be in [0, 1], "
                f"got {subject_patch_threshold}.")
        self.subject_patch_threshold = float(subject_patch_threshold)
        if subject_center_mode not in {'mean', 'median'}:
            raise ValueError(
                "subject_center_mode must be either 'mean' or 'median', "
                f"got {subject_center_mode!r}.")
        self.subject_center_mode = subject_center_mode
        if not 0.0 <= subject_visible_quadrant_ratio <= 1.0:
            raise ValueError(
                "subject_visible_quadrant_ratio must be in [0, 1], "
                f"got {subject_visible_quadrant_ratio}.")
        self.subject_visible_quadrant_ratio = float(subject_visible_quadrant_ratio)
        if subject_visible_outside_mode not in {'random', 'balanced'}:
            raise ValueError(
                "subject_visible_outside_mode must be either 'random' or 'balanced', "
                f"got {subject_visible_outside_mode!r}.")
        self.subject_visible_outside_mode = subject_visible_outside_mode
        if not 1 <= cross_layer_index <= depth:
            raise ValueError(f"cross_layer_index must be in [1, {depth}], got {cross_layer_index}.")
        if cross_layer_fusion_mode not in {'gated', 'residual'}:
            raise ValueError(
                "cross_layer_fusion_mode must be either 'gated' or 'residual', "
                f"got {cross_layer_fusion_mode!r}.")
        self.cross_layer_index = cross_layer_index
        self.cross_layer_fusion_mode = cross_layer_fusion_mode

        # MAE encoder specifics. Names match the original MAE implementation.
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.view_embed = nn.Parameter(torch.zeros(1, num_views, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # MAE decoder specifics.
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # Pretraining-only H6 -> decoder pathway. The gated mode lets the final
        # encoder feature control how much intermediate information is retained.
        self.cross_layer_norm = norm_layer(embed_dim)
        self.cross_layer_proj = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.cross_layer_gate = (
            nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
            if cross_layer_fusion_mode == 'gated' else None)
        self.cross_layer_dropout = nn.Dropout(cross_layer_dropout)
        self.cross_layer_gamma = nn.Parameter(torch.tensor(float(cross_layer_gamma_init)))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.context_pos_embed = nn.Parameter(torch.zeros(1, max(num_views - 1, 1), decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.context_pos_embed, std=.02)
        torch.nn.init.normal_(self.view_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def _center_quadrant_bias(self, length, view_ids, device):
        grid = int(length ** 0.5)
        assert grid * grid == length

        center_size = max(2, int(round(grid * self.center_ratio)))
        center_size = min(center_size, grid)
        start = (grid - center_size) // 2
        end = start + center_size
        mid = grid // 2

        yy = torch.arange(grid, device=device).view(grid, 1).expand(grid, grid)
        xx = torch.arange(grid, device=device).view(1, grid).expand(grid, grid)
        in_center = (yy >= start) & (yy < end) & (xx >= start) & (xx < end)

        quadrant = torch.full((grid, grid), -1, device=device, dtype=torch.long)
        quadrant[in_center & (yy < mid) & (xx < mid)] = 0
        quadrant[in_center & (yy < mid) & (xx >= mid)] = 1
        quadrant[in_center & (yy >= mid) & (xx < mid)] = 2
        quadrant[in_center & (yy >= mid) & (xx >= mid)] = 3
        quadrant = quadrant.reshape(1, length)

        target_quadrant = (view_ids % self.num_views).view(-1, 1)
        return (quadrant == target_quadrant).float()

    @torch.no_grad()
    def _subject_quadrant_visible(self, normalized_imgs, view_ids, length, len_keep,
                                  patch_subject_threshold=0.25, target_visible=0):
        """Choose visible patches from a crop-aligned fruit quadrant.

        The pretraining transform has already applied the independent random
        crop and flip when this method is called.  We therefore detect the
        fruit on the *transformed* image, estimate its foreground centroid,
        assign foreground patches to four body-relative quadrants, and keep a
        no visible patches in the view's target quadrant.  The remaining visible
        patches are sampled randomly outside that target quadrant.  This keeps
        the original 75% mask ratio while replacing only the mask strategy.

        The detector intentionally uses the same simple RGB yellow/saturation
        cue as the visualization: this is an unsupervised, training-time
        patch selector, not an additional learnable module.
        """
        if normalized_imgs.dim() != 4:
            raise ValueError("normalized_imgs must have shape [N, C, H, W].")
        n, _, height, width = normalized_imgs.shape
        grid = int(length ** 0.5)
        patch_size = self.patch_embed.patch_size[0]
        if grid * grid != length or height != grid * patch_size or width != grid * patch_size:
            raise ValueError("Subject-quadrant masking expects a square patch grid matching the image size.")

        imgs = normalized_imgs.float()
        mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        rgb = (imgs * std + mean).clamp(0.0, 1.0)
        max_rgb = rgb.amax(dim=1)
        min_rgb = rgb.amin(dim=1)
        yellow_score = (rgb[:, 0] + rgb[:, 1]) * 0.5 - rgb[:, 2]
        saturation = max_rgb - min_rgb
        brightness = rgb.mean(dim=1)
        foreground = (
            (yellow_score > 0.12)
            & (saturation > 0.12)
            & (brightness > 0.20)
        )

        # Convert pixel foreground to patch occupancy.  A patch is a subject
        # patch when at least 25% of its pixels satisfy the RGB cue.
        occupancy = F.avg_pool2d(
            foreground.float().unsqueeze(1), kernel_size=patch_size, stride=patch_size
        ).squeeze(1)
        subject_patch = occupancy >= float(patch_subject_threshold)

        # Body-relative centroid; samples with an empty detector fall back to
        # the image center and ordinary random masking remains well-defined.
        weight = foreground.float()
        yy = torch.arange(height, device=imgs.device, dtype=weight.dtype).view(1, height, 1)
        xx = torch.arange(width, device=imgs.device, dtype=weight.dtype).view(1, 1, width)
        mass = weight.sum(dim=(1, 2))
        safe_mass = mass.clamp_min(1.0)
        if self.subject_center_mode == 'median':
            # Coordinate-wise foreground median. Compared with the arithmetic
            # centroid, this is less sensitive to a thin shadow or other small
            # foreground tail near the lower crop boundary.
            half_mass = (mass * 0.5).unsqueeze(1)
            row_cumulative = weight.sum(dim=2).cumsum(dim=1)
            col_cumulative = weight.sum(dim=1).cumsum(dim=1)
            cy = (row_cumulative >= half_mass).to(torch.int64).argmax(dim=1).to(weight.dtype)
            cx = (col_cumulative >= half_mass).to(torch.int64).argmax(dim=1).to(weight.dtype)
        else:
            cy = (weight * yy).sum(dim=(1, 2)) / safe_mass
            cx = (weight * xx).sum(dim=(1, 2)) / safe_mass
        empty = mass <= 0
        cy = torch.where(empty, weight.new_full((), (height - 1) * 0.5), cy)
        cx = torch.where(empty, weight.new_full((), (width - 1) * 0.5), cx)

        patch_y = (torch.arange(grid, device=imgs.device, dtype=weight.dtype) + 0.5) * patch_size
        patch_x = (torch.arange(grid, device=imgs.device, dtype=weight.dtype) + 0.5) * patch_size
        py = patch_y.view(1, grid, 1)
        px = patch_x.view(1, 1, grid)
        qmap = torch.where(py < cy.view(-1, 1, 1),
                           torch.where(px < cx.view(-1, 1, 1), 0, 1),
                           torch.where(px < cx.view(-1, 1, 1), 2, 3))
        target_q = (view_ids.to(imgs.device) % self.num_views).view(-1, 1, 1)
        target = subject_patch & (qmap == target_q)
        target = target.reshape(n, length)
        subject_patch = subject_patch.reshape(n, length)

        target_count = target.sum(dim=1)
        reserve_count = target_count.clamp(max=int(target_visible))
        target_noise = torch.rand(n, length, device=imgs.device)
        target_rank = torch.argsort(torch.argsort(target_noise + (~target).float() * 2.0, dim=1), dim=1)
        reserved = target & (target_rank < reserve_count.unsqueeze(1))

        remaining = (len_keep - reserve_count).clamp_min(0)
        non_target = ~target
        non_target_count = non_target.sum(dim=1)
        take_non_target = torch.minimum(remaining, non_target_count)
        non_target_noise = torch.rand(n, length, device=imgs.device)
        non_target_rank = torch.argsort(
            torch.argsort(non_target_noise + target.float() * 2.0, dim=1), dim=1)
        selected_non_target = non_target & (non_target_rank < take_non_target.unsqueeze(1))

        leftover = remaining - take_non_target
        extra_target = target & (~reserved)
        extra_noise = torch.rand(n, length, device=imgs.device)
        extra_rank = torch.argsort(
            torch.argsort(extra_noise + (~extra_target).float() * 2.0, dim=1), dim=1)
        selected_extra_target = extra_target & (extra_rank < leftover.unsqueeze(1))
        selected = reserved | selected_non_target | selected_extra_target

        # Defensive exact-count fill for degenerate detector outputs.
        selected_count = selected.sum(dim=1)
        shortfall = (len_keep - selected_count).clamp_min(0)
        if (shortfall > 0).any():
            fallback_noise = torch.rand(n, length, device=imgs.device)
            fallback_rank = torch.argsort(
                torch.argsort(fallback_noise + selected.float() * 2.0, dim=1), dim=1)
            selected = selected | ((~selected) & (fallback_rank < shortfall.unsqueeze(1)))
        return selected

    @torch.no_grad()
    def _subject_quadrant_priority_visible(
            self, normalized_imgs, view_ids, length, len_keep,
            patch_subject_threshold=0.25, quadrant_visible_ratio=0.70,
            outside_mode='random'):
        """Allocate about 70% of visible tokens to one body-relative quadrant.

        Detected subject patches in the assigned quadrant are selected first.
        When they do not fill the quadrant quota, the remaining patches in the
        same quadrant are ranked by foreground occupancy.  The rest of the
        visible-token budget is sampled randomly outside the assigned quadrant.
        """
        if normalized_imgs.dim() != 4:
            raise ValueError("normalized_imgs must have shape [N, C, H, W].")
        n, _, height, width = normalized_imgs.shape
        grid = int(length ** 0.5)
        patch_size = self.patch_embed.patch_size[0]
        if grid * grid != length or height != grid * patch_size or width != grid * patch_size:
            raise ValueError(
                "Subject-quadrant visible masking expects a square patch grid matching the image size.")

        imgs = normalized_imgs.float()
        mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        rgb = (imgs * std + mean).clamp(0.0, 1.0)
        max_rgb = rgb.amax(dim=1)
        min_rgb = rgb.amin(dim=1)
        foreground = (
            (((rgb[:, 0] + rgb[:, 1]) * 0.5 - rgb[:, 2]) > 0.12)
            & ((max_rgb - min_rgb) > 0.12)
            & (rgb.mean(dim=1) > 0.20)
        )
        occupancy = F.avg_pool2d(
            foreground.float().unsqueeze(1),
            kernel_size=patch_size, stride=patch_size).squeeze(1)
        subject_patch = occupancy >= float(patch_subject_threshold)

        weight = foreground.float()
        yy = torch.arange(
            height, device=imgs.device, dtype=weight.dtype).view(1, height, 1)
        xx = torch.arange(
            width, device=imgs.device, dtype=weight.dtype).view(1, 1, width)
        mass = weight.sum(dim=(1, 2))
        safe_mass = mass.clamp_min(1.0)
        if self.subject_center_mode == 'median':
            half_mass = (mass * 0.5).unsqueeze(1)
            row_cumulative = weight.sum(dim=2).cumsum(dim=1)
            col_cumulative = weight.sum(dim=1).cumsum(dim=1)
            cy = (row_cumulative >= half_mass).to(torch.int64).argmax(dim=1).to(weight.dtype)
            cx = (col_cumulative >= half_mass).to(torch.int64).argmax(dim=1).to(weight.dtype)
        else:
            cy = (weight * yy).sum(dim=(1, 2)) / safe_mass
            cx = (weight * xx).sum(dim=(1, 2)) / safe_mass
        empty = mass <= 0
        cy = torch.where(empty, weight.new_full((), (height - 1) * 0.5), cy)
        cx = torch.where(empty, weight.new_full((), (width - 1) * 0.5), cx)

        patch_y = (
            torch.arange(grid, device=imgs.device, dtype=weight.dtype) + 0.5) * patch_size
        patch_x = (
            torch.arange(grid, device=imgs.device, dtype=weight.dtype) + 0.5) * patch_size
        py = patch_y.view(1, grid, 1)
        px = patch_x.view(1, 1, grid)
        qmap = torch.where(
            py < cy.view(-1, 1, 1),
            torch.where(px < cx.view(-1, 1, 1), 0, 1),
            torch.where(px < cx.view(-1, 1, 1), 2, 3))
        target_q = (view_ids.to(imgs.device) % self.num_views).view(-1, 1, 1)
        target_geometry = (qmap == target_q).reshape(n, length)
        subject_patch = subject_patch.reshape(n, length)
        occupancy = occupancy.reshape(n, length)

        quadrant_quota = int(round(len_keep * float(quadrant_visible_ratio)))
        quadrant_quota = max(0, min(len_keep, quadrant_quota))

        target_subject = target_geometry & subject_patch
        target_subject_count = target_subject.sum(dim=1).clamp(max=quadrant_quota)
        primary_noise = torch.rand(n, length, device=imgs.device)
        primary_rank = torch.argsort(torch.argsort(
            primary_noise + (~target_subject).float() * 2.0, dim=1), dim=1)
        selected = target_subject & (primary_rank < target_subject_count.unsqueeze(1))

        # Fill a short target-quadrant quota with the most fruit-like boundary
        # patches. Tiny random jitter provides reproducible tie-breaking.
        target_shortfall = (quadrant_quota - selected.sum(dim=1)).clamp_min(0)
        target_remaining = target_geometry & (~selected)
        boundary_score = -occupancy + (~target_remaining).float() * 2.0
        boundary_score = boundary_score + torch.rand_like(boundary_score) * 1.0e-6
        boundary_rank = torch.argsort(torch.argsort(boundary_score, dim=1), dim=1)
        selected = selected | (
            target_remaining & (boundary_rank < target_shortfall.unsqueeze(1)))

        # Fill the remaining budget outside the constrained quadrant.  The
        # balanced mode assigns equal quotas to the other three quadrants;
        # Random mode samples from the union of the remaining spatial regions.
        outside_needed = (len_keep - selected.sum(dim=1)).clamp_min(0)
        if outside_mode == 'balanced':
            qmap_flat = qmap.reshape(n, length)
            target_q_flat = target_q.reshape(n, 1)
            base_quota = torch.div(outside_needed, self.num_views - 1,
                                   rounding_mode='floor')
            remainder = outside_needed % (self.num_views - 1)
            for offset in range(1, self.num_views):
                source_q = (target_q_flat + offset) % self.num_views
                candidate = qmap_flat == source_q
                quota = base_quota + (remainder >= offset).to(base_quota.dtype)
                candidate_count = candidate.sum(dim=1)
                quota = torch.minimum(quota, candidate_count)
                noise = torch.rand(n, length, device=imgs.device)
                rank = torch.argsort(torch.argsort(
                    noise + (~candidate).float() * 2.0, dim=1), dim=1)
                selected = selected | (candidate & (rank < quota.unsqueeze(1)))
        else:
            outside = ~target_geometry
            outside_noise = torch.rand(n, length, device=imgs.device)
            outside_rank = torch.argsort(torch.argsort(
                outside_noise + target_geometry.float() * 2.0, dim=1), dim=1)
            selected = selected | (outside & (outside_rank < outside_needed.unsqueeze(1)))

        # Defensive exact-count fill for extreme off-center or empty detections.
        shortfall = (len_keep - selected.sum(dim=1)).clamp_min(0)
        if (shortfall > 0).any():
            fallback_noise = torch.rand(n, length, device=imgs.device)
            fallback_rank = torch.argsort(torch.argsort(
                fallback_noise + selected.float() * 2.0, dim=1), dim=1)
            selected = selected | ((~selected) & (fallback_rank < shortfall.unsqueeze(1)))
        return selected

    def random_masking(self, x, mask_ratio, view_ids=None, complementary=False,
                       mask_strategy='random', mask_source=None):
        """Per-sample masking. With complementary=True, each view masks a
        different center quadrant more aggressively while keeping randomness.

        ``mixed_3x3`` masks approximately half of the masked patches using
        non-overlapping 3x3 patch blocks and fills the remaining masked
        patches with independent random samples. The final mask count remains
        exactly ``int(length * mask_ratio)``.
        """
        n, length, dim = x.shape
        len_keep = int(length * (1 - mask_ratio))

        if mask_strategy == 'mixed_3x3':
            grid = int(length ** 0.5)
            if grid * grid != length or grid < 6:
                raise ValueError(
                    'mixed_3x3 requires a square patch grid with at least 6x6 patches.')
            num_mask = length - len_keep
            num_block_patches = min(num_mask, 9 * int(round((num_mask * 0.5) / 9)))
            num_block_patches = max(9, num_block_patches)
            num_blocks = num_block_patches // 9
            num_random = num_mask - num_block_patches

            # Randomly offset a 4x4 coarse tiling of 3x3 blocks, then choose
            # half of its cells. This guarantees non-overlapping blocks while
            # retaining a different spatial arrangement per sample.
            block_mask = torch.zeros((n, grid, grid), dtype=torch.bool, device=x.device)
            for sample_idx in range(n):
                row_offset = int(torch.randint(0, 3, (1,), device=x.device).item())
                col_offset = int(torch.randint(0, 3, (1,), device=x.device).item())
                chosen = torch.randperm(16, device=x.device)[:num_blocks]
                for block_idx in chosen.tolist():
                    row = row_offset + 3 * (block_idx // 4)
                    col = col_offset + 3 * (block_idx % 4)
                    block_mask[sample_idx, row:row + 3, col:col + 3] = True

            block_mask = block_mask.reshape(n, length)
            mask_noise = torch.rand(n, length, device=x.device)
            mask_noise = mask_noise.masked_fill(block_mask, float('inf'))
            random_ids = torch.argsort(mask_noise, dim=1)[:, :num_random]
            mixed_mask = block_mask.clone()
            mixed_mask.scatter_(1, random_ids, True)

            # Sort visible tokens first, with random order inside visible and
            # masked groups, matching the standard MAE gather/restore contract.
            order_noise = torch.rand(n, length, device=x.device)
            order_noise = order_noise + mixed_mask.float() * 2.0
            ids_shuffle = torch.argsort(order_noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(
                x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
            mask = mixed_mask.float()
            return x_masked, mask, ids_restore

        if mask_strategy == 'view_coverage':
            raise ValueError(
                'view_coverage masking must be used with token-fusion four-view input.')

        if mask_strategy == 'subject_quadrant':
            if view_ids is None or mask_source is None:
                raise ValueError('subject_quadrant masking requires view_ids and transformed images.')
            selected = self._subject_quadrant_visible(
                mask_source, view_ids, length, len_keep,
                patch_subject_threshold=self.subject_patch_threshold)
            order_noise = torch.rand(n, length, device=x.device)
            order_noise = order_noise + (~selected).float() * 2.0
            ids_shuffle = torch.argsort(order_noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(
                x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
            mask = torch.ones([n, length], device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
            return x_masked, mask, ids_restore

        if mask_strategy == 'subject_quadrant_visible':
            if view_ids is None or mask_source is None:
                raise ValueError(
                    'subject_quadrant_visible masking requires view_ids and transformed images.')
            selected = self._subject_quadrant_priority_visible(
                mask_source, view_ids, length, len_keep,
                patch_subject_threshold=self.subject_patch_threshold,
                quadrant_visible_ratio=self.subject_visible_quadrant_ratio,
                outside_mode=self.subject_visible_outside_mode)
            order_noise = torch.rand(n, length, device=x.device)
            order_noise = order_noise + (~selected).float() * 2.0
            ids_shuffle = torch.argsort(order_noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(
                x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
            mask = torch.ones([n, length], device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
            return x_masked, mask, ids_restore

        noise = torch.rand(n, length, device=x.device)
        if complementary and view_ids is not None:
            bias = self._center_quadrant_bias(length, view_ids.to(x.device), x.device)
            noise = noise + bias * self.complementary_bias

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        mask = torch.ones([n, length], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def view_coverage_masking(self, x, mask_ratio, shared_visible=4):
        """Jointly mask four views with low visible-token overlap.

        Each view keeps exactly ``len_keep`` patches. A small shared anchor
        set is visible in every view, while the remaining visible patches are
        allocated without replacement across views. With the default 224px
        input and 75% masking, this gives 4 shared + 45 unique patches per
        view and 184 distinct visible positions across the four views.
        """
        batch_size, num_views, length, dim = x.shape
        if num_views != self.num_views:
            raise ValueError(
                f'view_coverage expects {self.num_views} views, got {num_views}.')
        len_keep = int(length * (1 - mask_ratio))
        shared_visible = min(max(int(shared_visible), 0), len_keep)
        unique_visible = len_keep - shared_visible
        if shared_visible + num_views * unique_visible > length:
            raise ValueError(
                'view_coverage cannot allocate disjoint unique visible patches '
                f'for length={length}, views={num_views}, len_keep={len_keep}.')

        x_masked = torch.empty(
            (batch_size, num_views, len_keep, dim), device=x.device, dtype=x.dtype)
        mask = torch.ones(
            (batch_size, num_views, length), device=x.device, dtype=x.dtype)
        ids_restore = torch.empty(
            (batch_size, num_views, length), device=x.device, dtype=torch.long)

        for sample_idx in range(batch_size):
            permutation = torch.randperm(length, device=x.device)
            shared_ids = permutation[:shared_visible]
            remaining = permutation[shared_visible:]
            for view_idx in range(num_views):
                start = view_idx * unique_visible
                unique_ids = remaining[start:start + unique_visible]
                visible_ids = torch.cat([shared_ids, unique_ids], dim=0)
                visible_ids = visible_ids[torch.randperm(len_keep, device=x.device)]

                masked_selector = torch.ones(length, dtype=torch.bool, device=x.device)
                masked_selector[visible_ids] = False
                masked_ids = torch.arange(length, device=x.device)[masked_selector]
                ids_shuffle = torch.cat([visible_ids, masked_ids], dim=0)
                ids_restore[sample_idx, view_idx] = torch.argsort(ids_shuffle)
                mask[sample_idx, view_idx, visible_ids] = 0
                x_masked[sample_idx, view_idx] = torch.gather(
                    x[sample_idx, view_idx], dim=0,
                    index=visible_ids.unsqueeze(-1).expand(-1, dim))

        return (
            x_masked.reshape(batch_size * num_views, len_keep, dim),
            mask.reshape(batch_size * num_views, length),
            ids_restore.reshape(batch_size * num_views, length),
        )

    def forward_encoder(self, x, mask_ratio, view_ids=None, complementary=False,
                        mask_strategy='random', mask_source=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(
            x, mask_ratio, view_ids=view_ids, complementary=complementary,
            mask_strategy=mask_strategy, mask_source=mask_source)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_encoder_token_fusion(self, imgs, mask_ratio, use_complementary_mask=True,
                                     mask_strategy='random', return_intermediate=False):
        batch_size, num_views, channels, height, width = imgs.shape
        flat_imgs = imgs.reshape(batch_size * num_views, channels, height, width)

        x = self.patch_embed(flat_imgs)
        length = x.shape[1]
        x = x.reshape(batch_size, num_views, length, x.shape[-1])
        x = x + self.pos_embed[:, 1:, :].unsqueeze(1)
        x = x + self.view_embed[:, :num_views, :, :]

        view_ids = torch.arange(num_views, device=flat_imgs.device).repeat(batch_size)
        if mask_strategy == 'view_coverage':
            x, mask, ids_restore = self.view_coverage_masking(
                x, mask_ratio, shared_visible=4)
        else:
            x = x.reshape(batch_size * num_views, length, x.shape[-1])
            x, mask, ids_restore = self.random_masking(
                x, mask_ratio, view_ids=view_ids, complementary=use_complementary_mask,
                mask_strategy=mask_strategy, mask_source=flat_imgs)
        len_keep = x.shape[1]

        x = x.reshape(batch_size, num_views * len_keep, x.shape[-1])
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        intermediate = None
        for block_index, blk in enumerate(self.blocks, start=1):
            x = blk(x)
            if return_intermediate and block_index == self.cross_layer_index:
                intermediate = x
        x = self.norm(x)

        if return_intermediate:
            if intermediate is None:
                raise RuntimeError("Configured cross-layer feature was not collected.")
            return x, intermediate, mask, ids_restore, len_keep
        return x, mask, ids_restore, len_keep

    def _apply_context_dropout(self, context):
        if not self.training or self.context_dropout <= 0:
            return context
        keep = torch.rand(context.shape[:2], device=context.device).unsqueeze(-1)
        keep = (keep >= self.context_dropout).to(context.dtype)
        return context * keep

    def forward_decoder(self, x, ids_restore, context=None, cross_layer_latent=None):
        x = self.decoder_embed(x)

        if cross_layer_latent is not None:
            if cross_layer_latent.shape[:2] != x.shape[:2]:
                raise ValueError("Cross-layer and final encoder tokens must have matching shapes.")
            cross_layer = self.cross_layer_proj(self.cross_layer_norm(cross_layer_latent))
            deep_patch_tokens = x[:, 1:, :]
            mid_patch_tokens = self.cross_layer_dropout(cross_layer[:, 1:, :])
            if self.cross_layer_fusion_mode == 'gated':
                gate = torch.sigmoid(self.cross_layer_gate(deep_patch_tokens))
                residual = gate * mid_patch_tokens
            else:
                residual = mid_patch_tokens
            fused_patch_tokens = deep_patch_tokens + self.cross_layer_gamma * residual
            x = torch.cat([x[:, :1, :], fused_patch_tokens], dim=1)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x_ = x_ + self.decoder_pos_embed[:, 1:, :]

        cls_token = x[:, :1, :] + self.decoder_pos_embed[:, :1, :]
        if context is not None:
            context = self._apply_context_dropout(context)
            context = self.decoder_embed(context)
            context = context + self.context_pos_embed[:, :context.shape[1], :]
            x = torch.cat([cls_token, context, x_], dim=1)
            num_context = context.shape[1]
        else:
            x = torch.cat([cls_token, x_], dim=1)
            num_context = 0

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1 + num_context:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def _make_cross_view_context(self, latent, batch_size, num_views):
        latent = latent.reshape(batch_size, num_views, latent.shape[1], latent.shape[2])
        cls_tokens = latent[:, :, 0, :]
        contexts = []
        for view_idx in range(num_views):
            other_indices = [idx for idx in range(num_views) if idx != view_idx]
            contexts.append(cls_tokens[:, other_indices, :])
        context = torch.stack(contexts, dim=1)
        return context.reshape(batch_size * num_views, num_views - 1, cls_tokens.shape[-1])

    def forward_view_consistency_loss(self, latent, batch_size, num_views):
        """Encourage four views of the same pomelo to share one stable encoder representation."""
        latent = latent.reshape(batch_size, num_views, latent.shape[1], latent.shape[2])
        view_features = latent[:, :, 1:, :].mean(dim=2)
        view_features = F.normalize(view_features, dim=-1)
        pomelo_feature = F.normalize(view_features.mean(dim=1), dim=-1)
        cosine = (view_features * pomelo_feature.unsqueeze(1)).sum(dim=-1)
        return (1.0 - cosine).mean()

    def forward(self, imgs, mask_ratio=0.80,
                use_complementary_mask=False,
                mask_strategy=None,
                use_cross_view_context=False,
                use_token_fusion=True,
                use_view_consistency=False,
                view_consistency_weight=0.0,
                use_mvtf_cross_layer_fusion=False,
                return_loss_components=False):
        if mask_strategy is None:
            mask_strategy = 'random' if imgs.dim() == 4 else 'subject_quadrant_visible'
        if imgs.dim() == 4:
            latent, mask, ids_restore = self.forward_encoder(
                imgs, mask_ratio, mask_strategy=mask_strategy)
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_loss(imgs, pred, mask)
            if return_loss_components:
                zero = loss.new_zeros(())
                return loss, pred, mask, {
                    'mae_loss': loss.detach(),
                    'view_consistency_loss': zero,
                    'total_loss': loss.detach(),
                }
            return loss, pred, mask

        assert imgs.dim() == 5, "CMR pretraining expects [B, V, C, H, W] four-view batches."
        batch_size, num_views, channels, height, width = imgs.shape
        assert num_views == self.num_views, f"Expected {self.num_views} views, got {num_views}."

        flat_imgs = imgs.reshape(batch_size * num_views, channels, height, width)
        if use_token_fusion:
            if use_mvtf_cross_layer_fusion:
                latent, intermediate, mask, ids_restore, len_keep = self.forward_encoder_token_fusion(
                    imgs, mask_ratio, use_complementary_mask=use_complementary_mask,
                    mask_strategy=mask_strategy,
                    return_intermediate=True)
            else:
                latent, mask, ids_restore, len_keep = self.forward_encoder_token_fusion(
                    imgs, mask_ratio, use_complementary_mask=use_complementary_mask,
                    mask_strategy=mask_strategy)
                intermediate = None
            cls_tokens = latent[:, :1, :].unsqueeze(1).expand(-1, num_views, -1, -1)
            view_tokens = latent[:, 1:, :].reshape(batch_size, num_views, len_keep, latent.shape[-1])
            decoder_latent = torch.cat([cls_tokens, view_tokens], dim=2)
            decoder_latent = decoder_latent.reshape(batch_size * num_views, len_keep + 1, latent.shape[-1])
            decoder_intermediate = None
            if intermediate is not None:
                mid_cls_tokens = intermediate[:, :1, :].unsqueeze(1).expand(-1, num_views, -1, -1)
                mid_view_tokens = intermediate[:, 1:, :].reshape(
                    batch_size, num_views, len_keep, intermediate.shape[-1])
                decoder_intermediate = torch.cat([mid_cls_tokens, mid_view_tokens], dim=2)
                decoder_intermediate = decoder_intermediate.reshape(
                    batch_size * num_views, len_keep + 1, intermediate.shape[-1])
            pred = self.forward_decoder(
                decoder_latent, ids_restore, cross_layer_latent=decoder_intermediate)
            mae_loss = self.forward_loss(flat_imgs, pred, mask)

            view_consistency_loss = mae_loss.new_zeros(())
            loss = mae_loss
            if return_loss_components:
                return loss, pred, mask, {
                    'mae_loss': mae_loss.detach(),
                    'view_consistency_loss': view_consistency_loss.detach(),
                    'total_loss': loss.detach(),
                }
            return loss, pred, mask

        view_ids = torch.arange(num_views, device=flat_imgs.device).repeat(batch_size)

        latent, mask, ids_restore = self.forward_encoder(
            flat_imgs, mask_ratio, view_ids=view_ids,
            complementary=use_complementary_mask, mask_strategy=mask_strategy,
            mask_source=flat_imgs)

        context = None
        if use_cross_view_context:
            context = self._make_cross_view_context(latent, batch_size, num_views)

        pred = self.forward_decoder(latent, ids_restore, context=context)
        mae_loss = self.forward_loss(flat_imgs, pred, mask)
        view_consistency_loss = mae_loss.new_zeros(())
        if use_view_consistency and view_consistency_weight > 0:
            full_view_latent = self.forward_encoder_features(flat_imgs)
            view_consistency_loss = self.forward_view_consistency_loss(
                full_view_latent, batch_size, num_views)

        loss = mae_loss + view_consistency_weight * view_consistency_loss
        if return_loss_components:
            return loss, pred, mask, {
                'mae_loss': mae_loss.detach(),
                'view_consistency_loss': view_consistency_loss.detach(),
                'total_loss': loss.detach(),
            }
        return loss, pred, mask


def cmr_mae_vit_base_patch16_dec512d8b(**kwargs):
    model = CMRMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def cmr_mae_vit_large_patch16_dec512d8b(**kwargs):
    model = CMRMaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


cmr_mae_vit_base_patch16 = cmr_mae_vit_base_patch16_dec512d8b
cmr_mae_vit_large_patch16 = cmr_mae_vit_large_patch16_dec512d8b
