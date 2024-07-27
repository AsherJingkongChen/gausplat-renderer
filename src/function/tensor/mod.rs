use burn::tensor::{backend::Backend, Shape, Tensor};

pub trait TensorExtension {
    fn matmul_batched(
        self,
        other: Self,
    ) -> Self;
}

impl<B: Backend, const D: usize> TensorExtension for Tensor<B, D> {
    fn matmul_batched(
        self,
        rhs: Self,
    ) -> Self {
        const BATCH_SIZE: usize = (1 << 16) - 1;

        if D < 3 {
            return self.matmul(rhs);
        }

        let dim_l = self.dims();
        let dim_r = rhs.dims();
        let dim_lo = dim_l[0..D - 2].to_vec();
        let dim_ro = dim_r[0..D - 2].to_vec();
        let dim_l0 = dim_lo.iter().product::<usize>();
        let dim_r0 = dim_ro.iter().product::<usize>();
        let dim_l1 = dim_l[D - 2];
        let dim_r1 = dim_r[D - 2];
        let dim_l2 = dim_l[D - 1];
        let dim_r2 = dim_r[D - 1];
        assert_eq!(
            dim_l2, dim_r1,
            "The inner dimensions of matmul should be compatible. \
            We expected self.dims()[D - 1] == rhs.dims()[D - 2], \
            but got self.dims() = {:?} and rhs.dims() = {:?}",
            dim_l, dim_r
        );
        assert!(
            dim_l0 == dim_r0 || dim_l0 == 1 || dim_r0 == 1,
            "The outer dimensions of matmul should be compatible. \
            We expected self.dims()[0..D - 2] == rhs.dims()[0..D - 2] or some of them are all ones, \
            but got self.dims() = {:?} and rhs.dims() = {:?}",
            dim_l, dim_r
        );

        let (count, dim_o) = if dim_l0 != 1 {
            (dim_l0, dim_lo)
        } else {
            (dim_r0, dim_ro)
        };
        if count < BATCH_SIZE {
            return self.matmul(rhs);
        }

        let dims = Shape::from([dim_o, vec![dim_l1, dim_r2]].concat());
        let lhs = self.flatten::<3>(0, D - 3);
        let rhs = rhs.flatten::<3>(0, D - 3);

        Tensor::cat(
            (0..count)
                .step_by(BATCH_SIZE)
                .map(|index| {
                    let range = [index..(index + BATCH_SIZE).min(count)];
                    let lhs_batch = lhs.to_owned().slice(if dim_l0 == 1 {
                        [0..1]
                    } else {
                        range.to_owned()
                    });
                    let rhs_batch = rhs.to_owned().slice(if dim_r0 == 1 {
                        [0..1]
                    } else {
                        range.to_owned()
                    });

                    lhs_batch.matmul(rhs_batch)
                })
                .collect(),
            0,
        )
        .reshape(dims)
    }
}
