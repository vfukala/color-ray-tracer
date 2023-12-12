use num_complex::Complex;
use std::collections::HashMap;
use std::fs::read_to_string;
use std::ops;

type PositionValue = f32;
type WavelengthValue = f32;
type WavelengthIndex = usize;
type IntensityValue = f32;

const MIN_WL: WavelengthValue = 380e-9;
const MAX_WL: WavelengthValue = 750e-9;
const WLS_COUNT: usize = (750 - 380) / 5 + 1;

type Spectrum = [IntensityValue; WLS_COUNT];

fn index_to_wl(i: WavelengthIndex) -> WavelengthValue {
    MIN_WL + (MAX_WL - MIN_WL) / ((WLS_COUNT - 1) as WavelengthValue) * (i as WavelengthValue)
}

const EPS: PositionValue = 1e-4;

#[derive(Clone, Copy, Debug)]
struct RGBColor {
    data: [IntensityValue; 3],
}

impl RGBColor {
    const fn new(r: IntensityValue, g: IntensityValue, b: IntensityValue) -> Self {
        Self { data: [r, g, b] }
    }

    const fn black() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    const fn gray(v: IntensityValue) -> Self {
        Self::new(v, v, v)
    }
}

impl ops::Add<RGBColor> for RGBColor {
    type Output = Self;

    fn add(mut self, other: Self) -> RGBColor {
        for i in 0..3 {
            self.data[i] += other.data[i];
        }
        self
    }
}

impl ops::Mul<PositionValue> for RGBColor {
    type Output = Self;

    fn mul(mut self, c: PositionValue) -> Self {
        self.data.iter_mut().for_each(|d| *d *= c);
        self
    }
}

impl ops::Mul<RGBColor> for PositionValue {
    type Output = RGBColor;

    fn mul(self, v: RGBColor) -> RGBColor {
        v * self
    }
}

impl std::iter::Sum for RGBColor {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut total = RGBColor { data: [0.0; 3] };
        for v in iter {
            total = total + v;
        }
        total
    }
}

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    data: [PositionValue; 3],
}

impl Vec3 {
    const fn new(x: PositionValue, y: PositionValue, z: PositionValue) -> Self {
        Self { data: [x, y, z] }
    }

    const fn new_uniform(v: PositionValue) -> Self {
        Self::new(v, v, v)
    }

    const fn zero() -> Self {
        Self { data: [0.0; 3] }
    }

    fn dot(&self, other: &Self) -> PositionValue {
        let pairs = self.data.iter().zip(other.data.iter());
        pairs.map(|(&a, &b)| a * b).sum()
    }

    fn cross(&self, other: &Self) -> Self {
        Vec3 {
            data: (0..3)
                .map(|i| {
                    self.data[(i + 1) % 3] * other.data[(i + 2) % 3]
                        - self.data[(i + 2) % 3] * other.data[(i + 1) % 3]
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }

    fn length_sqr(&self) -> PositionValue {
        self.data.iter().map(|&a| a * a).sum::<PositionValue>()
    }

    fn length(&self) -> PositionValue {
        self.length_sqr().sqrt()
    }

    fn normalize(&mut self) {
        let length = self.length();
        for a in &mut self.data {
            *a /= length;
        }
    }

    fn normalized(&self) -> Self {
        let mut v = *self;
        v.normalize();
        v
    }

    // component-wise min
    fn cw_min(&self, other: &Self) -> Self {
        Vec3 {
            data: self.data.iter().zip(other.data.iter()).map(|(&v1, &v2)| v1.min(v2)).collect::<Vec<_>>().try_into().unwrap()
        }
    }

    // component-wise max
    fn cw_max(&self, other: &Self) -> Self {
        Vec3 {
            data: self.data.iter().zip(other.data.iter()).map(|(&v1, &v2)| v1.max(v2)).collect::<Vec<_>>().try_into().unwrap()
        }
    }
}

impl ops::Mul<PositionValue> for Vec3 {
    type Output = Self;

    fn mul(mut self, c: PositionValue) -> Self {
        self.data.iter_mut().for_each(|d| *d *= c);
        self
    }
}

impl ops::Mul<Vec3> for PositionValue {
    type Output = Vec3;

    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl ops::Add<Vec3> for Vec3 {
    type Output = Self;

    fn add(mut self, other: Self) -> Vec3 {
        for i in 0..3 {
            self.data[i] += other.data[i];
        }
        self
    }
}

impl ops::AddAssign<Vec3> for Vec3 {
    fn add_assign(&mut self, other: Self) {
        for i in 0..3 {
            self.data[i] += other.data[i];
        }
    }
}

impl ops::Sub<Vec3> for Vec3 {
    type Output = Self;

    fn sub(mut self, other: Self) -> Vec3 {
        for i in 0..3 {
            self.data[i] -= other.data[i];
        }
        self
    }
}

impl ops::Neg for Vec3 {
    type Output = Self;

    fn neg(mut self) -> Self {
        for i in 0..3 {
            self.data[i] = -self.data[i];
        }
        self
    }
}

impl std::iter::Sum for Vec3 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut total = Vec3 { data: [0.0; 3] };
        for v in iter {
            total = total + v;
        }
        total
    }
}

#[derive(Clone, Copy, Debug)]
struct Mat3 {
    data: [[PositionValue; 3]; 3], // row-major
}

impl Mat3 {
    const fn id() -> Self {
        Mat3 {
            data:
             [[1.0, 0.0, 0.0], 
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]]
        }
    }

    fn from_columns(columns: [Vec3; 3]) -> Self {
        Mat3 {
            data: (0..3).map(|i| (0..3).map(|j| columns[j].data[i]).collect::<Vec<_>>().try_into().unwrap()).collect::<Vec<_>>().try_into().unwrap()
        }
    }

    fn det(&self) -> PositionValue {
        (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| self.data[j][(i + j) % 3])
                    .product::<PositionValue>()
            })
            .sum::<PositionValue>()
            - (0..3)
                .map(|i| {
                    (0..3)
                        .map(|j| self.data[j][(i + 3 - j) % 3])
                        .product::<PositionValue>()
                })
                .sum::<PositionValue>()
    }

    fn solve(&self, b: &Vec3) -> Vec3 {
        let my_det = self.det();
        Vec3 {
            data: (0..3)
                .map(|i| {
                    let mut m = *self;
                    for j in 0..3 {
                        m.data[j][i] = b.data[j];
                    }
                    m.det() / my_det
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }

    fn transpose(&self) -> Self {
        Mat3 {
            data: (0..3).map(|i| (0..3).map(|j| self.data[j][i]).collect::<Vec<_>>().try_into().unwrap()).collect::<Vec<_>>().try_into().unwrap()
        }
    }
}

impl ops::Mul<PositionValue> for Mat3 {
    type Output = Self;

    fn mul(mut self, c: PositionValue) -> Self {
        self.data.iter_mut().for_each(|row|
                row.iter_mut().for_each(|d| *d *= c));
        self
    }
}

impl ops::Mul<Mat3> for PositionValue {
    type Output = Mat3;

    fn mul(self, m: Mat3) -> Mat3 {
        m * self
    }
}

// TODO add sample count per cell as an argument
// r1 and r2 should be normalized and either both 'to' or both 'from' the hit point
fn diffraction_intensity(
    b1: Vec3,
    b2: Vec3,
    r1: Vec3,
    r2: Vec3,
    wl: WavelengthValue,
    rad: PositionValue,
) -> IntensityValue {
    let bl1 = b1.length();
    let bl2 = b2.length();
    // we assume that b1 and b2 are ortogonal, so that
    // ||j*b1 + k*b2||= |j| * ||b1|| + |k| * ||b2||

    let dot1 = b1.dot(&(r1 + r2));
    let dot2 = b2.dot(&(r1 + r2));

    let mut count = 1;

    // 1.0 + 0.0 * i for the point at origin (exp(i * <0, rr>))
    let mut wave = Complex::<IntensityValue>::new(1.0, 0.0);

    let mut add_at = |j, k| {
        let spacial_dist = j * dot1 + k * dot2;
        let phase = std::f32::consts::FRAC_2_PI * spacial_dist / wl;
        wave += Complex::<IntensityValue>::cis(phase);
        count += 1;
    };

    {
        let mut j = 1.0;
        while j as PositionValue * bl1 <= rad {
            let mut k = 0.0;
            while j * bl1 + k * bl2 <= rad {
                add_at(j, k);
                add_at(k, -j);
                add_at(-j, -k);
                add_at(-k, j);
                k += 1.0;
            }
            j += 1.0;
        }
    }

    let i1 = (wave / count as IntensityValue).norm_sqr();

    let mut count2 = 1;

    // 1.0 + 0.0 * i for the point at origin (exp(i * <0, rr>))
    let mut wave2 = Complex::<IntensityValue>::new(0.0, 0.0);

    let mut add_at2 = |j, k| {
        let spacial_dist = j * dot1 + k * dot2;
        let phase = std::f32::consts::FRAC_2_PI * spacial_dist / wl;
        wave2 += Complex::<IntensityValue>::cis(phase);
        count2 += 1;
    };

    {
        let mut j = 0.5;
        while j * bl1 <= rad {
            let mut k = 0.5;
            while j * bl1 + k * bl2 <= rad {
                add_at2(j, k);
                add_at2(k, -j);
                add_at2(-j, -k);
                add_at2(-k, j);
                k += 1.0;
            }
            j += 1.0;
        }
    }
    let i2 = (wave2 / count2 as IntensityValue).norm_sqr();
    (i1 + i2) / 2.0
}

#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: Vec3,
    dir: Vec3,
}

impl Ray {
    fn at(&self, t: PositionValue) -> Vec3 {
        self.origin + t * self.dir
    }

    // express ray in a coordinate system (') for which
    // X = linmap * X' + translation
    // dir of the produced ray might not be normalized
    fn transform_into(&self, translation: &Vec3, linmap: &Mat3) -> Self {
        Ray {
            origin: linmap.solve(&(self.origin - *translation)),
            dir: linmap.solve(&self.dir),
        }
    }
}

#[derive(Debug)]
struct TracedRay {
    ray: Ray,
    ttl: i32,
}

#[derive(Debug)]
struct ShadowCheckRay {
    ray: Ray,
    light_t: PositionValue,
}

/*
struct Triangle<'a> {
    verts: [(Vec3, Vec3); 3], // array 3 of (position, normal)
    struct_center: Vec3,
    material: &'a dyn Material,
}

impl<'a> Triangle<'a> {
    fn vertex_matrix(&self) -> Mat3 {
        Mat3 {
            data: (0..3)
                .map(|i| {
                    (0..3)
                        .map(|j| self.verts[j].0.data[i])
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }

    fn bary(&self, pos: &Vec3) -> Vec3 {
        self.vertex_matrix().solve(pos)
    }

    fn normal_at(&self, bary: &Vec3) -> Vec3 {
        (0..3)
            .map(|i| bary.data[i] * self.verts[i].1)
            .sum::<Vec3>()
            .normalized()
    }

    fn normal_at_pos(&self, pos: &Vec3) -> Vec3 {
        self.normal_at(&self.bary(pos))
    }

    fn plane_normal(&self) -> Vec3 {
        (self.verts[1].0 - self.verts[0].0).cross(&(self.verts[2].0 - self.verts[0].0))
    }

    // TODO: if performance bad, don't recompute the barycentrics
    fn intersect(&self, ray: &Ray) -> Option<PositionValue> {
        let n = self.plane_normal();
        let c = n.dot(&(self.verts[0].0 - ray.origin));
        let t = c / n.dot(&ray.dir);
        if t < 0.0 {
            return None;
        }
        let bary = self.bary(&ray.at(t));
        if bary.data[0] >= 0.0 && bary.data[1] >= 0.0 && bary.data[2] >= 0.0 {
            Some(t)
        } else {
            None
        }
    }
}
*/

trait IntersectedObject {
    fn shade(
        &self,
        scene: &dyn Tracer,
        receptor: &dyn LightReceptor,
        in_lights: &mut dyn Iterator<Item = (Vec3, IntensityValue)>,
        ray_ttl: i32,
    ) -> RGBColor;
}

trait Object {
    fn simple_intersect(&self, ray: &Ray) -> Option<PositionValue>;
    fn intersect(&self, ray: &Ray) -> Option<(PositionValue, Box<dyn IntersectedObject + '_>)>;
}

struct CompactDiscObj {
    // geometry
    center: Vec3,
    normal: Vec3,
    r_inner: PositionValue,
    r_outer: PositionValue,

    // for vanilla specular shading
    diffusivity: IntensityValue,
    shininess: IntensityValue,

    // for reflection shading
    absorbtion: IntensityValue,

    // for diffraction shading
    diffraction_factor: IntensityValue,
    d_rad: PositionValue,
    d_perp: PositionValue,
    intg_rad: PositionValue,
    intg_samples_per_cell: u32,
}

impl CompactDiscObj {
    fn intersect_disc_plane(&self, ray: &Ray) -> (PositionValue, Vec3) {
        let t = self.normal.dot(&(self.center - ray.origin)) / self.normal.dot(&ray.dir);
        (t, ray.at(t))
    }
}

struct IntersectedCompactDiscObj<'a> {
    obj: &'a CompactDiscObj,
    hit_pos: Vec3,
    in_dir: Vec3,
}

impl Object for CompactDiscObj {
    fn simple_intersect(&self, ray: &Ray) -> Option<PositionValue> {
        let (t, hit_pos) = self.intersect_disc_plane(ray);
        let r_sqr = (hit_pos - self.center).length_sqr();
        if t >= 0.0 && self.r_inner * self.r_inner <= r_sqr && r_sqr <= self.r_outer * self.r_outer
        {
            Some(t)
        } else {
            None
        }
    }

    fn intersect(&self, ray: &Ray) -> Option<(PositionValue, Box<dyn IntersectedObject + '_>)> {
        let (t, hit_pos) = self.intersect_disc_plane(ray);
        let r_sqr = (hit_pos - self.center).length_sqr();
        if t >= 0.0 && self.r_inner * self.r_inner <= r_sqr && r_sqr <= self.r_outer * self.r_outer
        {
            Some((
                t,
                Box::new(IntersectedCompactDiscObj {
                    obj: self,
                    hit_pos,
                    in_dir: ray.dir,
                }),
            ))
        } else {
            None
        }
    }
}

impl<'a> IntersectedObject for IntersectedCompactDiscObj<'a> {
    fn shade(
        &self,
        tracer: &dyn Tracer,
        receptor: &dyn LightReceptor,
        in_lights: &mut dyn Iterator<Item = (Vec3, IntensityValue)>,
        ray_ttl: i32,
    ) -> RGBColor {
        let refl_color = if ray_ttl > 0 {
            let out_dir = self.in_dir - 2.0 * self.obj.normal.dot(&self.in_dir) * self.obj.normal;
            (1.0 - self.obj.absorbtion)
                * tracer.trace(&TracedRay {
                    ray: Ray {
                        dir: out_dir,
                        origin: self.hit_pos + EPS * out_dir,
                    },
                    ttl: ray_ttl - 1,
                })
        } else {
            RGBColor::black()
        };

        // for diffraction
        let r = self.hit_pos - self.obj.center;
        let b1 = self.obj.d_rad * r.normalized();
        let b2 = self.obj.d_perp * self.obj.normal.cross(&r).normalized();

        let own_color: RGBColor = in_lights
            .filter_map(|(dir, int)| {
                let n_dot = dir.dot(&self.obj.normal);
                if n_dot > 0.0 {
                    let h = (dir - self.in_dir).normalized();
                    let diffu = self.obj.diffusivity
                        * RGBColor::gray(self.obj.normal.dot(&h).powf(self.obj.shininess));
                    let spectrum = (0..WLS_COUNT)
                        .map(|wli| {
                            let wl = index_to_wl(wli);
                            // TODO add the number of samples parameter
                            diffraction_intensity(b1, b2, dir, -self.in_dir, wl, self.obj.intg_rad)
                        })
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap();
                    let diffra = self.obj.diffraction_factor * receptor.process(&spectrum);
                    Some(int * (diffu + diffra))
                } else {
                    None
                }
            })
            .sum();

        own_color + refl_color
    }
}

trait Tracer {
    fn trace(&self, tr: &TracedRay) -> RGBColor;
}

trait Light {
    fn illuminate(&self, pos: Vec3) -> Option<(ShadowCheckRay, IntensityValue)>;
}

struct DirectionalLight {
    dir: Vec3,
    intensity: IntensityValue,
}

impl Light for DirectionalLight {
    fn illuminate(&self, pos: Vec3) -> Option<(ShadowCheckRay, IntensityValue)> {
        Some((
            ShadowCheckRay {
                ray: Ray {
                    origin: pos - EPS * self.dir,
                    dir: -self.dir,
                },
                light_t: PositionValue::INFINITY,
            },
            self.intensity,
        ))
    }
}

/*
struct PointLight {
    pos: Vec3,
    coeffs: [IntensityValue; 3],
}
*/

struct Scene<'a> {
    objects: Vec<Box<dyn Object + 'a>>,
    lights: Vec<Box<dyn Light>>,
}

impl<'a> Scene<'a> {
    fn is_obstructed(&self, scr: &ShadowCheckRay) -> bool {
        self.objects.iter().any(|obj| {
            obj.simple_intersect(&scr.ray)
                .map_or(false, |t| t <= scr.light_t)
        })
    }
}

struct RTTracer<'a> {
    scene: &'a Scene<'a>,
    receptor: &'a dyn LightReceptor,
}

impl Tracer for RTTracer<'_> {
    fn trace(&self, tr: &TracedRay) -> RGBColor {
        let mut closest_t = f32::INFINITY;
        let mut closest: Option<Box<dyn IntersectedObject>> = None;

        for obj in &self.scene.objects {
            if let Some((t, iobj)) = obj.intersect(&tr.ray) {
                if t < closest_t {
                    closest_t = t;
                    closest = Some(iobj);
                }
            }
        }

        closest.map_or(RGBColor::black(), |iobj| {
            let hit_pos = tr.ray.at(closest_t);

            let mut in_lights = self.scene.lights.iter().filter_map(|l| {
                l.illuminate(hit_pos).and_then(|(scr, int)| {
                    if self.scene.is_obstructed(&scr) {
                        None
                    } else {
                        Some((scr.ray.dir, int))
                    }
                })
            });
            iobj.shade(self, self.receptor, &mut in_lights, tr.ttl)
        })
    }
}

/*
struct BoolTracer<'a> {
    scene: &'a Scene<'a>
}

impl Tracer for BoolTracer<'_> {
    fn trace(&self, tr: &TracedRay) -> IntensityValue {
        for tri in &self.scene.triangles {
            match tri.intersect(&tr.ray) {
                Some(_) => {
                    return 1.0
                }
                None => {}
            }
        }
        0.0
    }
}
*/

struct RawImage {
    pixels: Vec<RGBColor>,
    width: u32,
    height: u32,
}

trait LightReceptor {
    fn process(&self, spectrum: &Spectrum) -> RGBColor;
}

struct RTLightReceptor {
    receptors: [Spectrum; 3],
    factor: IntensityValue,
}

impl LightReceptor for RTLightReceptor {
    fn process(&self, spectrum: &Spectrum) -> RGBColor {
        RGBColor {
            data: (0..3)
                .map(|i| {
                    self.receptors[i]
                        .iter()
                        .zip(spectrum.iter())
                        .map(|(&rec, &sig)| rec * sig)
                        .sum::<IntensityValue>()
                        * self.factor
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}

trait Camera {
    fn render(&self, tracer: &dyn Tracer) -> RawImage;
}

struct RTCamera {
    pos: Vec3,
    view_dir: Vec3,

    // u and v form an orthogonal basis of the projection plane which is at distance 1 from the
    // camera's position
    proj_u: Vec3,
    proj_v: Vec3,

    proj_w: PositionValue,
    proj_h: PositionValue,
    pixel_w: u32,
    pixel_h: u32,
}

impl Camera for RTCamera {
    fn render(&self, tracer: &dyn Tracer) -> RawImage {
        let mut pixels = vec![RGBColor::black(); (self.pixel_w * self.pixel_h) as usize];

        for y in 0..self.pixel_h {
            for x in 0..self.pixel_w {
                let coeff_u = ((x as PositionValue + 0.5) / (self.pixel_w as PositionValue) - 0.5)
                    * self.proj_w;
                let coeff_v = ((y as PositionValue + 0.5) / (self.pixel_h as PositionValue) - 0.5)
                    * self.proj_h;
                let ray = Ray {
                    origin: self.pos,
                    dir: (self.view_dir + coeff_u * self.proj_u + coeff_v * self.proj_v)
                        .normalized(),
                };
                pixels[(y * self.pixel_h + x) as usize] = tracer.trace(&TracedRay { ray, ttl: 2 })
            }
        }

        RawImage {
            pixels,
            width: self.pixel_w,
            height: self.pixel_h,
        }
    }
}

trait ImageWriter {
    fn write_image(image: RawImage, filename: &str);
}

struct RTImageWriter {}

impl ImageWriter for RTImageWriter {
    fn write_image(image: RawImage, filename: &str) {
        let bytes = image
            .pixels
            .iter()
            .flat_map(|&rgb| {
                rgb.data
                    .iter()
                    .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        image::save_buffer(
            filename,
            bytes.as_slice(),
            image.width,
            image.height,
            image::ColorType::Rgb8,
        )
        .unwrap();
    }
}

fn load_light_receptor(filename: &str) -> RTLightReceptor {
    let rec_values = read_to_string(filename)
        .unwrap()
        .lines()
        .map(|s| {
            let parts = s.split(',').collect::<Vec<_>>();
            (
                parts[0].parse::<i32>().unwrap(),
                (0..3)
                    .map(|i| parts[i + 1].parse::<IntensityValue>().unwrap())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            )
        })
        .collect::<HashMap<i32, [IntensityValue; 3]>>();
    RTLightReceptor {
        receptors: (0..3)
            .map(|i| {
                (0..WLS_COUNT)
                    .map(|wli| {
                        let wlnm = (index_to_wl(wli) * 1e9).round() as i32;
                        rec_values.get(&wlnm).unwrap()[i]
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
        factor: 10.0,
    }
}

fn construct_scene<'a>(bunny_mesh: &'a Mesh) -> Scene {
    let cd1 = CompactDiscObj {
        center: Vec3::new(-0.6, 0.0, 0.0),
        normal: Vec3::new(1.0, 0.0, -1.0),
        r_inner: 0.2,
        r_outer: 0.7,
        diffusivity: 0.1,
        shininess: 5.0,
        absorbtion: 0.1,
        diffraction_factor: 0.5,
        d_rad: 1.5e-6 * 2.0,
        d_perp: 3.0e-6 * 2.0,
        intg_rad: 50.0e-6,
        intg_samples_per_cell: 1,
    };
    let cd2 = CompactDiscObj {
        center: Vec3::new(0.6, 0.0, 0.0),
        normal: Vec3::new(-1.0, 0.0, -1.0),
        r_inner: 0.2,
        r_outer: 0.7,
        diffusivity: 0.1,
        shininess: 5.0,
        absorbtion: 0.1,
        diffraction_factor: 0.5,
        d_rad: 1.5e-6 * 2.0,
        d_perp: 3.0e-6 * 2.0,
        intg_rad: 50.0e-6,
        intg_samples_per_cell: 1,
    };
    let bunny = MeshObj {
        mesh: bunny_mesh,
        mesh_aabb: bunny_mesh.aabb(),
        pos: Vec3::new(-0.01234, 0.01876, -0.5),
        linmap: 4.0 * Mat3::id(),
        ambient_color: RGBColor::new(0.09, 0.02, 0.02),
        diffuse_color: RGBColor::new(0.9, 0.2, 0.2),
    };
    let main_light = DirectionalLight {
        dir: Vec3::new(0.3, -0.2, 4.0).normalized(),
        intensity: 1.0,
    };
    Scene {
        objects: vec![Box::new(cd1), Box::new(cd2), Box::new(bunny)],
        lights: vec![Box::new(main_light)],
    }
}

fn construct_tracer<'a>(scene: &'a Scene, receptor: &'a dyn LightReceptor) -> Box<dyn Tracer + 'a> {
    Box::new(RTTracer { scene, receptor })
}

fn construct_camera() -> RTCamera {
    RTCamera {
        pos: Vec3::new(0.0, 0.0, -6.0),
        view_dir: Vec3::new(0.0, 0.0, 1.0),
        proj_u: Vec3::new(1.0, 0.0, 0.0),
        proj_v: Vec3::new(0.0, -1.0, 0.0),
        proj_w: 0.4,
        proj_h: 0.4,
        pixel_w: 1024,
        pixel_h: 1024,
    }
}

struct MeshVertex {
    pos: Vec3,
    normal: Vec3,
}

struct Mesh {
    vertices: Vec<MeshVertex>,
    faces: Vec<[ u32; 3 ]>
}

fn load_ply(filename: &str, center: bool) -> Mesh {
    let content = read_to_string(filename).unwrap();
    let lines = content.lines();
    let mut vcount_lines = lines.skip_while(|s| !s.starts_with("element vertex "));
    let vcount = vcount_lines.next().unwrap()["element vertex ".len()..].parse::<usize>().unwrap();
    let mut fcount_lines = vcount_lines.skip_while(|s| !s.starts_with("element face "));
    let fcount = fcount_lines.next().unwrap()["element face ".len()..].parse::<usize>().unwrap();
    let mut data_lines = fcount_lines.skip_while(|s| !s.starts_with("end_header")).skip(1);

    let pre_vert_pos = (0..vcount).map(|_| {
        let s = data_lines.next().unwrap();
        Vec3 {
            data: s.split(' ').take(3).map(|num_s| num_s.parse::<PositionValue>().unwrap()).collect::<Vec<_>>().try_into().unwrap()
        }
    }).collect::<Vec<_>>();

    let vert_pos = if center {
        let mins = pre_vert_pos.iter().fold(Vec3::new_uniform(PositionValue::INFINITY), |state, pos| state.cw_min(&pos));
        let maxs = pre_vert_pos.iter().fold(Vec3::new_uniform(PositionValue::NEG_INFINITY), |state, pos| state.cw_max(&pos));
        let mids = 0.5 * (mins + maxs);
        pre_vert_pos.into_iter().map(|pos| pos - mids).collect::<Vec<_>>()
    } else {
        pre_vert_pos
    };

    let faces = (0..fcount).map(|_| {
        let s = data_lines.next().unwrap();
        s.split(' ').skip(1).take(3).map(|num_s| num_s.parse::<u32>().unwrap()).collect::<Vec<_>>().try_into().unwrap()
    }).collect::<Vec<_>>();
    let face_normals = faces.iter().map(|f: &[_; 3]|
                                        (vert_pos[f[1] as usize] - vert_pos[f[0] as usize])
                                        .cross(&(vert_pos[f[2] as usize] - vert_pos[f[0] as usize])).normalized()).collect::<Vec<_>>();
    let mut vertex_normal_sums = vec! [ Vec3::zero(); vcount ];
    for i in 0..fcount {
        for j in 0..3 {
            let vi = faces[i][j] as usize;
            vertex_normal_sums[vi] += face_normals[i];
        }
    }
    Mesh {
        vertices: vert_pos.into_iter().zip(vertex_normal_sums.into_iter()).map(|(pos, norm_sum)| MeshVertex { pos, normal: norm_sum.normalized() }).collect::<Vec<_>>(),
        faces,
    }
}

struct MeshObj<'a> {
    mesh: &'a Mesh,
    mesh_aabb: AABB,
    pos: Vec3,
    linmap: Mat3,
    ambient_color: RGBColor,
    diffuse_color: RGBColor,
}

impl<'a> MeshObj<'a> {
    fn ray_to_local(&self, ray: &Ray) -> Ray {
        ray.transform_into(&self.pos, &self.linmap)
    }

    fn maybe_intersects(&self, local_ray: &Ray) -> bool {
        // check if rays hits the bounding box
        let t_maxmin = (0..3).map(|i| {
            let ray_compo = local_ray.dir.data[i];
            PositionValue::min(
                (self.mesh_aabb.mins.data[i] - local_ray.origin.data[i]) / ray_compo, 
                (self.mesh_aabb.maxs.data[i] - local_ray.origin.data[i]) / ray_compo 
                )
        }).fold(PositionValue::NEG_INFINITY, |state, v| state.max(v));
        let t_minmax = (0..3).map(|i| {
            let ray_compo = local_ray.dir.data[i];
            PositionValue::max(
                (self.mesh_aabb.mins.data[i] - local_ray.origin.data[i]) / ray_compo, 
                (self.mesh_aabb.maxs.data[i] - local_ray.origin.data[i]) / ray_compo 
                )
        }).fold(PositionValue::INFINITY, |state, v| state.min(v));
        t_maxmin <= t_minmax
    }
}

#[derive(Debug)]
struct AABB {
    mins: Vec3,
    maxs: Vec3,
}

impl Mesh {
    fn aabb(&self) -> AABB {
        AABB {
            mins: self.vertices.iter().fold(Vec3::new_uniform(PositionValue::INFINITY), |state, vert| state.cw_min(&vert.pos)),
            maxs: self.vertices.iter().fold(Vec3::new_uniform(PositionValue::NEG_INFINITY), |state, vert| state.cw_max(&vert.pos)),
        }
    }
}

impl<'a> Object for MeshObj<'a> {
    fn simple_intersect(&self, ray: &Ray) -> Option<PositionValue> {
        let local_ray = self.ray_to_local(ray);

        if !self.maybe_intersects(&local_ray) {
            return None;
        }

        let mut closest_t = PositionValue::INFINITY;
        for f in &self.mesh.faces {
            let tri: [Vec3; 3] = (0..3).map(|i| self.mesh.vertices[f[i] as usize].pos).collect::<Vec<_>>().try_into().unwrap();
            let n = (tri[1] - tri[0]).cross(&(tri[2] - tri[0]));
            let t = n.dot(&(tri[0] - local_ray.origin)) / n.dot(&local_ray.dir);
            if t < 0.0 || t >= closest_t {
                continue;
            }
            let hit_pos = local_ray.at(t);
            let vm = Mat3::from_columns(tri);
            let bary = vm.solve(&hit_pos);
            if bary.data[0] >= 0.0 && bary.data[1] >= 0.0 && bary.data[2] >= 0.0 {
                closest_t = t;
            }
        }
        if closest_t == PositionValue::INFINITY {
            None
        } else {
            Some(closest_t)
        }
    }

    fn intersect(&self, ray: &Ray) -> Option<(PositionValue, Box<dyn IntersectedObject + '_>)> {
        let local_ray = self.ray_to_local(ray);

        if !self.maybe_intersects(&local_ray) {
            return None;
        }

        let mut closest_t = PositionValue::INFINITY;
        let mut closest_face_i = 0;
        let mut closest_bary = Vec3::zero();
        for i in 0..self.mesh.faces.len() {
            let f = self.mesh.faces[i];
            let tri: [Vec3; 3] = (0..3).map(|i| self.mesh.vertices[f[i] as usize].pos).collect::<Vec<_>>().try_into().unwrap();
            let n = (tri[1] - tri[0]).cross(&(tri[2] - tri[0]));
            let t = n.dot(&(tri[0] - local_ray.origin)) / n.dot(&local_ray.dir);
            if t < 0.0 || t >= closest_t {
                continue;
            }
            let hit_pos = local_ray.at(t);
            let vm = Mat3::from_columns(tri);
            let bary = vm.solve(&hit_pos);
            if bary.data[0] >= 0.0 && bary.data[1] >= 0.0 && bary.data[2] >= 0.0 {
                closest_t = t;
                closest_face_i = i;
                closest_bary = bary;
            }
        }
        if closest_t == PositionValue::INFINITY {
            None
        } else {
            Some((closest_t, Box::new(IntersectedMeshObj {
                obj: self,
                face_i: closest_face_i,
                bary: closest_bary,
            })))
        }
    }
}

struct IntersectedMeshObj<'a> {
    obj: &'a MeshObj<'a>,
    face_i: usize,
    bary: Vec3,
}

impl<'a> IntersectedObject for IntersectedMeshObj<'a> {
    fn shade(
        &self,
        _scene: &dyn Tracer,
        _receptor: &dyn LightReceptor,
        in_lights: &mut dyn Iterator<Item = (Vec3, IntensityValue)>,
        _ray_ttl: i32,
    ) -> RGBColor {
        let mesh_space_n = self.obj.mesh.faces[self.face_i].iter().zip(self.bary.data.iter())
            .map(|(&vi, &bar)| bar * self.obj.mesh.vertices[vi as usize].normal).sum::<Vec3>().normalized();
        let n = self.obj.linmap.transpose().solve(&mesh_space_n).normalized();
        in_lights
            .filter_map(|(dir, int)| {
                let n_dot = dir.dot(&n);
                if n_dot > 0.0 {
                    Some(int * n.dot(&dir))
                } else {
                    None
                }
            })
            .sum::<IntensityValue>() * self.obj.diffuse_color + self.obj.ambient_color
    }
}

fn main() {
    let receptor = load_light_receptor("receptor.csv");
    let bunny_mesh = load_ply("../../bun_zipper_res3.ply", true);
    let scene = construct_scene(&bunny_mesh);
    let tracer = construct_tracer(&scene, &receptor);
    let camera = construct_camera();
    let image = camera.render(tracer.as_ref());
    RTImageWriter::write_image(image, "out.png");
}
