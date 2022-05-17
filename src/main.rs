use ndarray::prelude::*;
use ndarray::{stack, Array1, Array2};
// use ndarray_linalg::Inverse;

const NO: usize = 800; // 样本量
const NY: usize = 9; // Indicator数量
const NK: usize = 3; // 潜在自变量（XI or Fx） + 潜在因变量（Eta）个数
const ND: usize = 2; // Eta的协变量
const NM: usize = 1; // 潜在因变量
const NZ: usize = 2; // 潜自变量
const NG: usize = 3; // 潜在自变量 + 非线性项
const NB: usize = 6; // 协变量 + eta + [潜在自变量 + 潜在因变量] （在蔡老师程序里把这个合并为一个矩阵）
const NH: usize = 5;
const BN: usize = 1;
const NP: usize = 76; // 参数的个数

const MCAX: usize = 10000;
const GNUM: usize = 5000;
const CNUM: usize = 1;
const SS: usize = 1;

/// 将BI的值切片分配给BD，PII和PB
fn rel1(BI: &Array2<f64>, BD: &mut Array2<f64>, PII: &mut Array2<f64>, PB: &mut Array2<f64>) {
    BD.assign(&BI.slice(s![.., ..ND]));
    PII.assign(&BI.slice(s![.., ND..ND + NM]));
    PB.assign(&BI.slice(s![.., ND + NM..]));
}

fn gxi2(gx2: &mut Array2<f64>, x2: &Array2<f64>) {}

fn rela2(
    GOmega: &Array2<f64>,
    BGZ: &Array2<f64>,
    GEta: &mut Array2<f64>,
    GXI: &mut Array2<f64>,
    GFx: &Array2<f64>,
    GXIB: &mut Array2<f64>,
) {
    GEta.assign(&GOmega.slice(s![.., ..NM]));
    GXI.assign(&GOmega.slice(s![.., NM..NM + NZ]));
}

fn genBZ(BZ: &mut Array2<f64>) {
    // The equivalent logic to genBZ.c
    for i in 0..NO {
        BZ[[i, 0]] = 0.0;
        BZ[[i, 1]] = 1.0;
    }
}

fn gene(YO: &mut Array2<f64>, XO: &mut Array2<f64>, BZ: &mut Array2<f64>, TUE: &mut Array2<f64>) {
    // The equivalent logic to read.c
    // Just read true.txt to variables
    let LY = array![
        [1.0, 0.0, 0.0],
        [0.9, 0.0, 0.0],
        [0.9, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.8, 0.0],
        [0.0, 0.8, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.7],
        [0.0, 0.0, 0.7],
    ];
    let MU = Array1::<f64>::zeros(NY);
    let PSX = array![0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 1., 1., 1.];
    let BD = array![[0.3, 0.3]];
    let PII = array![[0.]];
    // 潜在自变量的系数矩阵（包括非线性项）
    let PB = array![[0.8, -0.5, 0.2]];
    let PSD = array![0.25];
    let PHI = array![[1.0, 0.5], [0.5, 1.0],];

    genBZ(BZ);
}

fn main() {
    // read1()
    let IND = array![2., 2., 2., 2., 2., 2., 0., 0., 0.,];
    let ROAL = Array1::<f64>::ones(NY);
    let IDY = array![
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 0., 1.],
    ];
    let IDB = array![[1., 1., 0., 1., 1., 1.]];

    // read5()
    // 非线性项（实际只需取下三角或者上三角就好了）
    let NON = array![[0., 1.], [1., 0.]];

    // read7()
    let PALPA = Array1::from_elem(NY, 9.0);
    let PBETA = Array1::from_elem(NY, 4.0);
    let PALP = array![9.];
    let PBE = array![4.];
    let RHO = 4;
    let RH = Array2::<f64>::eye(NZ);
    let sigmu = 1.0;
    let sigly = 1.0;
    let sigbi = 1.0;

    // read8()
    let VAR_xi = 2.0;
    let VAR_mu = 10.0;
    let VAR_ly = 10.0;
    let VAR_alp = 0.01;

    for CIR in 1..=CNUM {
        // gene();

        // read3()
        // 载荷矩阵 𝜆
        let mut LY = array![
            [1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 0., 0.],
            [0., 0., 0.],
        ];
        let mut PLY = LY.clone();
        let mut MU = Array1::<f64>::zeros(NY);
        let mut PMU = Array1::<f64>::zeros(NY);
        let mut PSX = Array1::<f64>::ones(NY);

        // read4()
        let mut BI = Array2::<f64>::zeros((NM, NB));
        let mut PBI = Array2::<f64>::zeros((NM, NB));
        let mut PSD = array![1.0];
        let mut PHI = Array2::<f64>::eye(NZ);
        // let mut INPH = PHI.inv().unwrap();

        let mut BD = Array2::<f64>::zeros((NM, ND));
        let mut PII = Array2::<f64>::zeros((NM, NM));
        let mut PB = Array2::<f64>::zeros((NM, NG));
        rel1(&BI, &mut BD, &mut PII, &mut PB);

        let mut EOmega = Array2::<f64>::zeros((NO, NK));
        let mut EPA = Array1::<f64>::zeros(NP);
        let mut VPA = EPA.clone();
        let mut PA = EPA.clone();

        let Accept_Omega = 0.0;
        let Accept_ALPHA = 0.0;
        let Accept_MU = 0.0;
        let Accept_LY = 0.0;

        let mut Omega = EOmega.clone();
        for i in 0..NO {}

        /* Gibbson Sample*/
        for GIB in 1..=MCAX {
            for i in 0..NO {}

            rel1(&BI, &mut BD, &mut PII, &mut PB);

            if GIB % 100 == 0 {
                println!("{}", GIB);
            }

            if GIB > GNUM && GIB % SS == 0 {
                EOmega += &Omega;
                EPA += &PA;
                VPA += &PA.mapv(|a: f64| a.powi(2));
            }
        }
        let tmp: f64 = (SS / (MCAX - GNUM)) as f64;
        EOmega.mapv_inplace(|a: f64| a * tmp);
        EPA.mapv_inplace(|a: f64| a * tmp);
        VPA.zip_mut_with(&EPA, |a: &mut f64, b: &f64| {
            *a = ((*a) * tmp - (*b).powi(2)).sqrt()
        });
        let AVAC: f64 = Accept_Omega / (MCAX - NO) as f64;
        println!("Average Acceptance Rate of Omega: {}", AVAC);
    }
}

#[cfg(test)]
mod test {
    use ndarray::AssignElem;

    use super::*;

    use crate::{NK, NY};

    #[test]
    fn test_inverse() {
        let a = Array2::<f64>::eye(2);
        assert_eq!(a, array![[1., 0.], [0., 1.]]);
        // let ia = a.inv().unwrap();
        // println!("{:?}", ia);
    }

    #[test]
    fn test_map() {
        let mut a = Array2::<f64>::zeros((NY, NK));
        let mut b = Array2::<f64>::ones((NY, NK));
        a.mapv_inplace(|x| x + 1.);
        assert_eq!(a, Array2::from_elem((NY, NK), 1.));

        b.map_inplace(|x| *x = (*x).sqrt());
        assert_eq!(a, Array2::from_elem((NY, NK), 1.));

        a += &b.mapv(|x| x * 2. + 2.);
        assert_eq!(a, Array2::from_elem((NY, NK), 5.));

        println!("{:?}", b);
        println!("{:?}", a);

        a.zip_mut_with(&b, |a: &mut f64, b: &f64| {
            *a = ((*a) * 2. - (*b).powi(2)).sqrt()
        });
        println!("{:?}", a);
        assert_eq!(a, Array2::from_elem((NY, NK), 3.));
    }

    #[test]
    fn test_rel1() {
        let BI = array![[1., 2., 3., 4., 5., 6.]];
        let mut BD = Array2::<f64>::zeros((NM, ND));
        let mut PII = Array2::<f64>::zeros((NM, NM));
        let mut PB = Array2::<f64>::zeros((NM, NG));
        rel1(&BI, &mut BD, &mut PII, &mut PB);

        PB.slice_mut(s![.., ..ND]).assign(&BD);

        println!("{}", BD);
        println!("{}", PII);
        println!("{}", PB);
        println!("{}", BI);

        let x = stack![Axis(0), BD, BD];
        println!("{:?}", x);

        let a: f64 = 10.;
        let b = a.sqrt();
        println!("{}", b);
    }
}
