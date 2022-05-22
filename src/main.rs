use std::ops::AddAssign;

use ndarray::{prelude::*, AssignElem, Slice};
use ndarray::{stack, Array1, Array2};
use ndarray_linalg::cholesky::*;
use ndarray_linalg::{Cholesky, Inverse, Solve};
use rand::{thread_rng, Rng};
use rand_distr::{Standard, StandardNormal};

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

fn gxi2(gx2: &mut Array2<f64>, x2: &Array2<f64>) {
    gx2.slice_axis_mut(Axis(1), Slice::from(..NZ)).assign(x2);
    let _col0 = gx2.slice_axis(Axis(1), Slice::from(..1));
    let _col1 = gx2.slice_axis(Axis(1), Slice::from(1..2));
    let x = _col0.to_owned() * _col1.to_owned();
    gx2.slice_axis_mut(Axis(1), Slice::from(NZ..NG)).assign(&x);
}

fn rela2(
    GOmega: &Array2<f64>,
    GBZ: &Array2<f64>,
    GEta: &mut Array2<f64>,
    GXI: &mut Array2<f64>,
    GFx: &mut Array2<f64>,
    GXIB: &mut Array2<f64>,
) {
    GEta.assign(&GOmega.slice(s![.., ..NM]));
    GXI.assign(&GOmega.slice(s![.., NM..NM + NZ]));
    GXIB.slice_axis_mut(Axis(1), Slice::from(..ND)).assign(GBZ);
    GXIB.slice_axis_mut(Axis(1), Slice::from(ND..ND + NM))
        .assign(GEta);
    gxi2(GFx, GXI);
    GXIB.slice_axis_mut(Axis(1), Slice::from(ND + NM..ND + NM + NG))
        .assign(GFx);
}

fn genBZ(BZ: &mut Array2<f64>) {
    // The equivalent logic to genBZ.c
    for i in 0..NO {
        let mut rng = thread_rng();
        BZ[[i, 0]] = match rng.gen_bool(0.6) {
            true => 1.0,
            false => 0.0,
        };
        BZ[[i, 1]] = rng.sample(StandardNormal);
    }
}

fn fac(PHI: &Array2<f64>, XI: &mut Array2<f64>) {
    let PH = PHI.cholesky(UPLO::Lower).unwrap();
    let mut rand = Array1::<f64>::zeros(NZ);
    XI.map_axis_mut(Axis(1), |mut a| {
        rand.map_inplace(|b| *b = thread_rng().sample(StandardNormal));
        let r = PH.dot(&rand);
        a.add_assign(&r);
    });
}

fn init(
    BD: &Array2<f64>,
    PB: &Array2<f64>,
    PHI: &Array2<f64>,
    PII: &Array2<f64>,
    PSD: &Array1<f64>,
    BZ: &Array2<f64>,
    Omega: &mut Array2<f64>,
) {
    let mut Fx = Array2::<f64>::zeros((NO, NG));
    let mut fma = Array2::<f64>::eye(NM);
    let mut XI = Array2::<f64>::zeros((NO, NZ));
    fac(PHI, &mut XI);
    fma -= PII;
    Omega
        .slice_axis_mut(Axis(1), Slice::from(NM..NM + NZ))
        .assign(&XI);

    let mut delta = Array1::<f64>::zeros(NM);
    delta.zip_mut_with(PSD, |a, b| {
        let tmp: f64 = thread_rng().sample(StandardNormal);
        *a = tmp * (*b).sqrt();
    });
    gxi2(&mut Fx, &XI);
    let tmp = BZ.dot(&BD.t());
    delta += &tmp.sum_axis(Axis(0));
    let tmp = Fx.dot(&PB.t());
    delta += &tmp.sum_axis(Axis(0));
    let result = fma.solve_into(delta).unwrap();
    Omega
        .slice_axis_mut(Axis(1), Slice::from(..NM))
        .assign(&result);
}

fn gey(
    LY: &Array2<f64>,
    MU: &Array1<f64>,
    PSX: &Array1<f64>,
    TUE: &Array2<f64>,
    YO: &mut Array2<f64>,
) {
    let tmp_matrix = TUE.dot(&(LY.t()));
    for i in 0..NO {
        for j in 0..NY {
            let tmp: f64 = thread_rng().sample(StandardNormal);
            YO[[i, j]] = tmp * PSX[j].sqrt() + MU[j];
        }
    }
    YO.zip_mut_with(&tmp_matrix, |a, b| *a += *b);
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
    let PHI = array![[1.0, 0.5], [0.5, 1.0]];

    genBZ(BZ);

    init(&BD, &PB, &PHI, &PII, &PSD, BZ, TUE);

    gey(&LY, &MU, &PSX, TUE, YO);

    XO.assign(YO);
}

/// calculate Sigma_omega
fn calsm(
    LY: &Array2<f64>,
    MU: &Array1<f64>,
    PII: &Array2<f64>,
    PB: &Array2<f64>,
    INPH: &Array2<f64>,
    PSD: &Array1<f64>,
    PSX: &Array1<f64>,
    SIG: &mut Array2<f64>,
    VAR: f64,
) {
    /* II=I-PII */
    let mut II = Array2::<f64>::eye(NM);
    II -= PII;

    /* PH=PHI^{-1}+PB'*\Psi_\delta^{-1}*PB */
    let mut PH = INPH.clone();
    for i in 0..NZ {
        for j in 0..i {
            let _col1 = PB.column(i);
            let _col2 = PB.column(j);
            let _col3 = _col2.to_owned() / PSD;
            PH[[i, j]] += _col1.t().dot(&_col3);
            PH[[j, i]] = PH[[i, j]];
        }
    }

    /* SIG[1:NM-1][1:NM-1]=Pi_0'*\Psi_\delta^{-1}*Pi_0 */
    for i in 0..NM {
        for j in 0..i {
            let _col1 = II.column(i);
            let _col2 = II.column(j);
            let _col3 = _col2.to_owned() / PSD;
            SIG[[i, j]] = _col1.t().dot(&_col3);
            SIG[[j, i]] = SIG[[i, j]];
        }
    }

    /* SIG[NM:NK-1][NM:NK-1]=PH */
    SIG.slice_mut(s![.., NM..NK]).assign(&PH);

    for i in 0..NM {
        for j in NM..NM + NZ {
            let _col1 = II.column(i);
            let _col2 = II.column(j);
            let _col3 = _col2.to_owned() / PSD;
            SIG[[i, j]] = _col1.t().dot(&_col3);
            SIG[[j, i]] = SIG[[i, j]];
        }
    }

    for i in 0..NK {
        for j in 0..i {
            let _col1 = LY.column(i);
            let _col2 = LY.column(j);
            let _col3 = _col2.to_owned() / PSX;
            SIG[[i, j]] += _col1.t().dot(&_col3);
            PH[[i, j]] = SIG[[i, j]] / VAR;
            PH[[j, i]] = PH[[i, j]];
        }
    }

    SIG.assign(&PH.inv().unwrap());
}

fn tomega(
    GN: &Array2<f64>,
    GBZ: &Array2<f64>,
    GYO: &Array2<f64>,
    MU: &Array1<f64>,
    LY: &Array2<f64>,
    INPH: &Array2<f64>,
    BD: &Array2<f64>,
    PII: &Array2<f64>,
    PB: &Array2<f64>,
    PSD: &Array1<f64>,
    PSX: &Array1<f64>,
) -> Array1<f64> {
    let mut f2: Array1<f64>;
    let mut f3: Array1<f64>;

    /* measurement equation */
    let mut temp = GYO - MU;
    temp -= &GN.dot(&LY.t());
    temp.map_inplace(|a| *a = a.powi(2) * 0.5);
    let f1 = temp.sum_axis(Axis(1)) / PSX;

    /* exogenous latent variable */
    let mut GXI = Array2::<f64>::zeros((NO, NZ));
    GXI.assign(&GN.slice(s![.., NM..NM + NZ]));
    let f2 = GXI.map_axis(Axis(1), |a| {
        let mut f = 0.;
        for i in 0..NZ {
            f += a[i] * INPH[[i, i]] * a[i];
            for j in 0..i {
                f += a[i] * INPH[[i, j]] * a[j] * 2.;
            }
        }
        f
    });

    let mut Fx = Array2::<f64>::zeros((NO, NG));
    gxi2(&mut Fx, &GXI);

    let mut temp = Array2::<f64>::zeros((NO, NM));
    temp.assign(&GN.slice(s![.., ..NM]));
    temp -= &GBZ.dot(&BD.t());
    temp -= &GN.slice(s![.., ..NM]).dot(&PII.t());
    temp -= &Fx.dot(&PB.t());
    temp.mapv_inplace(|a| a.powi(2));
    let f3 = (temp / PSD).sum_axis(Axis(1));

    f1 + 0.5 * (f2 + f3)
}

fn genomega(
    GOmega: &mut Array2<f64>,
    GYO: &Array2<f64>,
    GBZ: &Array2<f64>,
    MU: &Array1<f64>,
    LY: &Array2<f64>,
    INPH: &Array2<f64>,
    BD: &Array2<f64>,
    PII: &Array2<f64>,
    PB: &Array2<f64>,
    PSD: &Array1<f64>,
    PSX: &Array1<f64>,
    ISG: &Array2<f64>,
) -> Array1<f64> {
    /* PP is distributed from N(GOmega,ISG) */
    let mut PP = GOmega.clone();
    let mut diag = Array2::<f64>::zeros((NK, NO));
    let PH = ISG.cholesky(UPLO::Lower).unwrap();
    diag.map_inplace(|a| *a = thread_rng().sample(StandardNormal));
    PP += &diag.dot(&PH);

    let rat1 = tomega(GOmega, GBZ, GYO, MU, LY, INPH, BD, PII, PB, PSD, PSX);
    let rat2 = tomega(&PP, GBZ, GYO, MU, LY, INPH, BD, PII, PB, PSD, PSX);
    let ratio = (rat1 - rat2).mapv(f64::ln);
    GOmega.assign(&PP);
    ratio
}

fn chan(
    MU: &Array1<f64>,
    LY: &Array2<f64>,
    PHI: &Array2<f64>,
    BI: &Array2<f64>,
    PSD: &Array1<f64>,
    PSX: &Array1<f64>,
    ALPHA: &Array2<f64>,
    PA: &mut Array1<f64>,
) {
    PA.slice_mut(s!(..NY)).assign(MU);
}

fn main() {
    let mut ALPHA = Array2::<f64>::zeros((NY, NH));
    let mut ISG = Array2::<f64>::zeros((NK, NK));
    let mut SIMG = Array2::<f64>::zeros((NK, NK));
    let mut SIMB = Array2::<f64>::zeros((NB, NB));
    // read1() read ind.txt
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

    // read5() read non.txt
    // 非线性项（实际只需取下三角或者上三角就好了）
    let mut NON = Array2::<isize>::zeros((NG, 9));
    NON[[2, 0]] = 1;
    NON[[2, 1]] = 2;
    NON[[2, 2]] = -1;

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
        /* gene(YO, XO, BZ, TUE); */
        let mut BZ = Array2::<f64>::zeros((NO, ND));
        let mut TUE = Array2::<f64>::zeros((NO, NK));
        let mut YO = Array2::<f64>::zeros((NO, NY));
        let mut XO = YO.clone();
        gene(&mut YO, &mut XO, &mut BZ, &mut TUE);

        // read3(LY, MU, PMU, PLY, PSX, ALPHA);
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

        /* read4(BI, PBI, PHI, PSD); */
        let mut BI = Array2::<f64>::zeros((NM, NB));
        let mut PBI = Array2::<f64>::zeros((NM, NB));
        let mut PSD = array![1.0];
        let mut PHI = Array2::<f64>::eye(NZ);

        /* rel1(BI, BD, PII, PB); */
        let mut BD = Array2::<f64>::zeros((NM, ND));
        let mut PII = Array2::<f64>::zeros((NM, NM));
        let mut PB = Array2::<f64>::zeros((NM, NG));
        rel1(&BI, &mut BD, &mut PII, &mut PB);

        /* get the inverse of PHI */
        let mut INPH = PHI.inv().unwrap();

        let mut EOmega = Array2::<f64>::zeros((NO, NK));
        let mut EPA = Array1::<f64>::zeros(NP);
        let mut VPA = EPA.clone();
        let mut PA = EPA.clone();

        let mut Accept_Omega = 0;
        let Accept_ALPHA = 0;
        let Accept_MU = 0;
        let Accept_LY = 0;

        let mut Omega = EOmega.clone();

        let mut XIB = Array2::<f64>::zeros((NO, NB));
        let mut XI = Array2::<f64>::zeros((NO, NZ));
        let mut Eta = Array2::<f64>::zeros((NO, NM));
        let mut Fx = Array2::<f64>::ones((NO, NG));
        rela2(&Omega, &BZ, &mut Eta, &mut XI, &mut Fx, &mut XIB);

        for i in 0..NO {}

        /* Gibbson Sample*/
        for GIB in 1..=MCAX {
            calsm(&LY, &MU, &PII, &PB, &INPH, &PSD, &PSX, &mut ISG, VAR_xi);
            let mut GOmega = Omega.clone();
            let RATIO = genomega(
                &mut GOmega,
                &YO,
                &BZ,
                &MU,
                &LY,
                &INPH,
                &BD,
                &PII,
                &PB,
                &PSD,
                &PSX,
                &ISG,
            );

            for i in 0..NO {
                let unfm: f64 = thread_rng().sample(Standard);
                if unfm <= RATIO[i] {
                    Omega.slice_mut(s![i, ..]).assign(&GOmega.slice(s![i, ..]));
                    Accept_Omega += 1;
                }
            }

            rela2(&Omega, &BZ, &mut Eta, &mut XI, &mut Fx, &mut XIB);

            // newph(PHI, INPH, XI, RHO, RH);

            for i in 0..NK {
                let _col = Omega.column(i);
                let _row = _col.t();
                for j in 0..i {
                    SIMG[[i, j]] += _row.dot(&Omega.column(j));
                    SIMG[[j, i]] = SIMG[[i, j]]
                }
            }

            for i in 0..NB {
                let _col = XIB.column(i);
                let _row = _col.t();
                for j in 0..i {
                    SIMG[[i, j]] += _row.dot(&XIB.column(j));
                    SIMG[[j, i]] = SIMG[[i, j]]
                }
            }

            // newmu();

            // newly();

            // newbi();

            chan(&MU, &LY, &PHI, &BI, &PSD, &PSX, &ALPHA, &mut PA);

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
        let AVAC: f64 = Accept_Omega as f64 / (MCAX - NO) as f64;
        println!("Average Acceptance Rate of Omega: {}", AVAC);

        println!("{:?}", EPA);
        println!("{:?}", VPA);
    }
}

#[cfg(test)]
mod test {
    use ndarray::{AssignElem, Slice};

    use super::*;

    use crate::{NK, NY};

    #[test]
    fn test_inverse() {
        let a = Array2::<f64>::eye(2);
        assert_eq!(a, array![[1., 0.], [0., 1.]]);
        let ia = a.inv().unwrap();
        println!("{:?}", ia);
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

        // PB.slice_mut(s![.., ..ND]).assign(&BD);
        PB.slice_axis_mut(Axis(1), Slice::from(..ND)).assign(&BD);

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

    #[test]
    fn test_rand() {
        let mut bz = Array2::<f64>::zeros((NO, 2));
        genBZ(&mut bz);
        println!("{:?}", bz);

        let mut gx = array![[1., 2., 0.], [1., 2., 0.], [1., 2., 0.], [1., 2., 0.]];
        {
            let col1 = gx.slice_axis(Axis(1), Slice::from(..1));
            println!("{:?}", col1);
            let col2 = gx.slice_axis(Axis(1), Slice::from(1..2));
            println!("{:?}", col2);
            let x = col1.to_owned() * col2.to_owned();
            println!("{:?}", x);
            let mut y = gx.slice_axis_mut(Axis(1), Slice::from(-1..));
            println!("{:?}", y);
            y.assign(&x);
        }
        println!("{:?}", gx);
    }

    #[test]
    fn test_gey() {
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
        let mut TUE = Array2::<f64>::zeros((NO, NK));
        let mut YO = Array2::<f64>::zeros((NO, NY));
        gey(&LY, &MU, &PSX, &TUE, &mut YO);
        println!("{:?}", YO);
        let mut PA = Array1::<f64>::zeros(NP);
        PA.slice_mut(s!(..NY)).assign(&PSX);
        println!("{:?}", PA);
    }

    #[test]
    fn test_cholsky() {
        let PHI = array![[1.0, 0.5], [0.5, 1.0],];
        let PH = PHI.cholesky(UPLO::Lower).unwrap();
        println!("{:?}", PH);
        let MU = array![0.1, 0.2];
        println!("{:?}", PHI - MU);
    }

    #[test]
    fn test_fac() {
        let PHI = array![[1.0, 0.5], [0.5, 1.0]];
        let mut XI = Array2::<f64>::zeros((NO, NZ));
        fac(&PHI, &mut XI);
        println!("{:?}", XI);
        let r = PHI.column(0).t().dot(&PHI.column(1));
        println!("{:?}", r);
    }

    #[test]
    fn test_init() {
        let BD = array![[0.3, 0.3]];
        let PII = array![[0.]];
        let PB = array![[0.8, -0.5, 0.2]];
        let PSD = array![0.25];
        let PHI = array![[1.0, 0.5], [0.5, 1.0]];
        let mut BZ = Array2::<f64>::zeros((NO, ND));
        let mut TUE = Array2::<f64>::zeros((NO, NK));
        genBZ(&mut BZ);
        println!("{:?}", BZ);

        init(&BD, &PB, &PHI, &PII, &PSD, &BZ, &mut TUE);
        println!("{:?}", TUE);
    }
}
