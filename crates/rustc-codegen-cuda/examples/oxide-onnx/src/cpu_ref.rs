/*!
 * CPU reference implementations for unit-test verification.
 *
 * These are pure Rust, use ndarray where helpful, and produce reference
 * outputs to validate each CUDA kernel in isolation.
 */


// ============================================================================
// Element-wise ops
// ============================================================================

pub fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

pub fn clip(x: &[f32], lo: f32, hi: f32) -> Vec<f32> {
    x.iter().map(|&v| v.clamp(lo, hi)).collect()
}

pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&a, &b)| a + b).collect()
}

// ============================================================================
// GEMM: C = alpha * A * B + beta * C   (row-major, m×k × k×n = m×n)
// ============================================================================

pub fn sgemm(
    m: usize, n: usize, k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &[f32],
) -> Vec<f32> {
    let mut out = c.to_vec();
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += a[row * k + i] * b[i * n + col];
            }
            out[row * n + col] = alpha * sum + beta * out[row * n + col];
        }
    }
    out
}

// ============================================================================
// Bias add
// ============================================================================

pub fn bias_add(x: &[f32], bias: &[f32], features: usize) -> Vec<f32> {
    x.iter().enumerate().map(|(i, &v)| v + bias[i % features]).collect()
}

// ============================================================================
// BatchNorm (inference)
// ============================================================================

pub fn batch_norm_inference(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
    c: usize,
    hw: usize,
) -> Vec<f32> {
    x.iter().enumerate().map(|(i, &xi)| {
        let chan = (i / hw) % c;
        gamma[chan] * (xi - mean[chan]) / (var[chan] + eps).sqrt() + beta[chan]
    }).collect()
}

// ============================================================================
// Softmax (row-wise, stable)
// ============================================================================

pub fn softmax(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        let base = row * cols;
        let max_val = x[base..base + cols].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..cols {
            let e = (x[base + j] - max_val).exp();
            out[base + j] = e;
            sum += e;
        }
        for j in 0..cols {
            out[base + j] /= sum;
        }
    }
    out
}

// ============================================================================
// MaxPool2D
// ============================================================================

pub fn maxpool2d(
    x: &[f32],
    n: usize, c: usize, in_h: usize, in_w: usize,
    kh: usize, kw: usize,
    pad_h: usize, pad_w: usize,
    stride_h: usize, stride_w: usize,
) -> (Vec<f32>, usize, usize) {
    let out_h = (in_h + 2 * pad_h).saturating_sub(kh) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w).saturating_sub(kw) / stride_w + 1;
    let mut out = vec![f32::NEG_INFINITY; n * c * out_h * out_w];

    for bn in 0..n {
        for bc in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = oh * stride_h + ki;
                            let iw = ow * stride_w + kj;
                            let ih_unpad = ih as isize - pad_h as isize;
                            let iw_unpad = iw as isize - pad_w as isize;
                            if ih_unpad >= 0 && ih_unpad < in_h as isize
                                && iw_unpad >= 0 && iw_unpad < in_w as isize
                            {
                                let v = x[bn * c * in_h * in_w
                                    + bc * in_h * in_w
                                    + ih_unpad as usize * in_w
                                    + iw_unpad as usize];
                                if v > max_val { max_val = v; }
                            }
                        }
                    }
                    out[bn * c * out_h * out_w + bc * out_h * out_w + oh * out_w + ow] = max_val;
                }
            }
        }
    }
    (out, out_h, out_w)
}

// ============================================================================
// Global Average Pool
// ============================================================================

pub fn global_avg_pool(x: &[f32], n: usize, c: usize, hw: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * c];
    for bn in 0..n {
        for bc in 0..c {
            let base = bn * c * hw + bc * hw;
            out[bn * c + bc] = x[base..base + hw].iter().sum::<f32>() / hw as f32;
        }
    }
    out
}

// ============================================================================
// Conv2D (direct: not im2col, used only for small unit tests)
// ============================================================================

pub fn conv2d(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    n: usize, c_in: usize, h_in: usize, w_in: usize,
    n_out: usize, kh: usize, kw: usize,
    pad_h: usize, pad_w: usize,
    stride_h: usize, stride_w: usize,
) -> (Vec<f32>, usize, usize) {
    let out_h = (h_in + 2 * pad_h).saturating_sub(kh) / stride_h + 1;
    let out_w = (w_in + 2 * pad_w).saturating_sub(kw) / stride_w + 1;
    let mut out = vec![0.0f32; n * n_out * out_h * out_w];

    for bn in 0..n {
        for co in 0..n_out {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for ci in 0..c_in {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh * stride_h + ki;
                                let iw = ow * stride_w + kj;
                                let ih_unpad = ih as isize - pad_h as isize;
                                let iw_unpad = iw as isize - pad_w as isize;
                                if ih_unpad >= 0 && ih_unpad < h_in as isize
                                    && iw_unpad >= 0 && iw_unpad < w_in as isize
                                {
                                    let x_val = x[bn * c_in * h_in * w_in
                                        + ci * h_in * w_in
                                        + ih_unpad as usize * w_in
                                        + iw_unpad as usize];
                                    let w_val = w[co * c_in * kh * kw
                                        + ci * kh * kw
                                        + ki * kw
                                        + kj];
                                    sum += x_val * w_val;
                                }
                            }
                        }
                    }
                    if let Some(b) = bias {
                        sum += b[co];
                    }
                    out[bn * n_out * out_h * out_w + co * out_h * out_w + oh * out_w + ow] = sum;
                }
            }
        }
    }
    (out, out_h, out_w)
}

// ============================================================================
// Test utility: max absolute difference
// ============================================================================

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// Return indices of the top-k largest values.
pub fn top_k(x: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = x.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.iter().take(k).map(|(i, _)| *i).collect()
}
