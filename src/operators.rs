use crate::tensor::Tensor;

/// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert_eq!(table_shape.len(), 2);
    let dim = table_shape[1];
    assert_eq!(y.size(), length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

/// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert_eq!(shape.len(), 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // todo!("实现 rms_norm，计算前做一些必要的检查会帮助你后续调试")
    assert_eq!(y.shape(), x.shape());
    assert_eq!(*y.shape().last().unwrap(), w.size());

    let dim = w.size(); // n
    let batch = x.size() / dim; // len/n
    let y_ = unsafe { y.data_mut() };

    for i in 0..batch {
        let x_i = x.slice(i * dim, w.shape());

        let coefficient = x_i.data().iter().map(|t| t * t).sum::<f32>() / (dim as f32) + epsilon;
        let coefficient = coefficient.sqrt();

        for (j, (&x_ij, w_j)) in x_i.data().iter().zip(w.data().iter()).enumerate() {
            y_[i * dim + j] = x_ij * w_j / coefficient;
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert_eq!(len, x.size());

    let y_ = unsafe { y.data_mut() };
    let x_ = x.data();

    for (i, x_i) in x_.iter().enumerate() {
        y_[i] = sigmoid(*x_i) * y_[i] * x_i;
    }
}

// hint: You don't need to do an explicit transpose of B
/// C = beta * C + alpha * A @ B^T
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");
    assert_eq!(a.shape().last().unwrap(), b.shape().last().unwrap());
    assert_eq!(c.shape(), &vec![a.shape()[0], b.shape()[0]]);

    let dim = *a.shape().last().unwrap(); // A 和 B 的共享维度
    let m = a.shape()[0]; // A 的行数
    let n = b.shape()[0]; // B 的行数

    let c_ = unsafe { c.data_mut() };

    for c_i in c_.iter_mut() {
        *c_i *= beta;
    }

    for i in 0..m {
        for j in 0..n {
            let a_slice = a.slice(i * dim, &vec![dim]);
            let b_slice = b.slice(j * dim, &vec![dim]);
            let dot_product = dot(&a_slice, &b_slice);

            c_[i * n + j] += alpha * dot_product;
        }
    }
}

/// A = A + B
pub fn add(a: &mut Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape());
    let a_ = unsafe { a.data_mut() };
    let b_ = b.data();
    for i in 0..a_.len() {
        a_[i] += b_[i];
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert_eq!(len, y.size());
    // let x_ = x.data();
    // let y_ = y.data();
    // let mut sum = 0.0;
    // for i in 0..len {
    //     sum += x_[i] * y_[i];
    // }
    // sum
    x.data()
        .iter()
        .zip(y.data().iter())
        .map(|(i, j)| i * j)
        .sum()
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert_eq!(x.shape()[x.shape().len() - 1], x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3,
    ));
}

#[test]
fn test_add() {
    let mut a=Tensor::<f32>::new(vec![1.,2.,3.,4.], &vec![2,2]);
    let b=Tensor::<f32>::new(vec![1.,2.,3.,4.], &vec![2,2]);
    add(&mut a, &b);
    assert!(a.close_to(
        &Tensor::<f32>::new(
            vec![2.,4.,6.,8.], 
            &vec![2,2]), 
        1e-3));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2],
        ),
        1e-3,
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3,
    ));
}
