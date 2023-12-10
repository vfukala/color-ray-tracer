use std::ops;
use std::collections::HashMap;
use num_complex::Complex;
// use num_traits::identities::Zero;

type PositionValue = f32;
type WavelengthValue = f32;
type WavelengthIndex = usize;
type IntensityValue = f32;
type RGBColor = [IntensityValue; 3];

const MIN_WL: WavelengthValue = 380e-9;
const MAX_WL: WavelengthValue = 750e-9;
//const WL_STEP: WavelengthValue = 10e-9;
const WLS_COUNT: usize = (750 - 380) / 5 + 1;

type Spectrum = [IntensityValue; WLS_COUNT];

fn index_to_wl(i: WavelengthIndex) -> WavelengthValue {
    MIN_WL + (MAX_WL - MIN_WL) / ((WLS_COUNT - 1) as WavelengthValue) * (i as WavelengthValue)
}

const EPS: PositionValue = 1e-4;

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    data: [PositionValue; 3],
}

impl Vec3 {
    fn new(x: PositionValue, y: PositionValue, z: PositionValue) -> Self {
        Self {
            data: [ x, y, z ]
        }
    }

    fn zero() -> Self {
        Self {
            data: [ 0.0; 3 ]
        }
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

    fn length(&self) -> PositionValue {
        self.data
            .iter()
            .map(|&a| a * a)
            .sum::<PositionValue>()
            .sqrt()
    }

    fn normalize(&mut self) {
        let length = self.length();
        for a in self.data.iter_mut() {
            *a /= length;
        }
    }

    fn normalized(&self) -> Self {
        let mut v = *self;
        v.normalize();
        v
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
            self.data[i] += other.data[i]
        }
        self
    }
}

impl ops::Sub<Vec3> for Vec3 {
    type Output = Self;

    fn sub(mut self, other: Self) -> Vec3 {
        for i in 0..3 {
            self.data[i] -= other.data[i]
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
    fn det(&self) -> PositionValue {
        let res = (0..3)
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
                .sum::<PositionValue>();
        res
    }

    fn solve(&self, b: &Vec3) -> Vec3 {
        let my_det = self.det();
        Vec3 {
            data: (0..3)
                .map(|i| {
                    let mut m = *self;
                    for j in 0..3 {
                        m.data[j][i] = b.data[j]
                    }
                    m.det() / my_det
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}

// r1 and r2 should be normalized and either both 'to' or both 'from' the hit point
fn diffraction_intensity(b1: Vec3, b2: Vec3, r1: Vec3, r2: Vec3, wl: WavelengthValue, rad: PositionValue) -> IntensityValue {
    let bl1 = b1.length();
    let bl2 = b2.length();
    // we assume that b1 and b2 are ortogonal, so that
    // ||j*b1 + k*b2||= |j| * ||b1|| + |k| * ||b2||
    
    let dot1 = b1.dot(&(r1 + r2));
    let dot2 = b2.dot(&(r1 + r2));
    
    let mut count = 1;

    // TODO: due to symmetry (+/-), the wave is always real;
    // so we can only compute cos instead of cis

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

    // TODO: due to symmetry (+/-), the wave is always real;
    // so we can only compute cos instead of cis

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
}

#[derive(Debug)]
struct TracedRay {
    ray: Ray,
    ttl: i32,
    wli: WavelengthIndex,
}

#[derive(Debug)]
struct ShadowCheckRay {
    ray: Ray,
    light_t: PositionValue,
}

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

// TODO: if performance bad, devirtualize by "going the lookup myself" (have a fixed list of
// materials and a big if-else)
trait Material {
    fn hit(
        &self,
        tri: &Triangle,
        hit_pos: &Vec3,
        tr: &TracedRay,
        in_lights: &mut dyn Iterator<Item = (Vec3, IntensityValue)>,
        scene: &dyn Tracer,
    ) -> IntensityValue;
}

struct MatteMaterial {
    color: Spectrum,
}

impl Material for MatteMaterial {
    fn hit(
        &self,
        tri: &Triangle,
        hit_pos: &Vec3,
        tr: &TracedRay,
        in_lights: &mut dyn Iterator<Item = (Vec3, IntensityValue)>,
        _scene: &dyn Tracer,
    ) -> IntensityValue {
        let n = tri.normal_at_pos(hit_pos);
        let total_light: IntensityValue = in_lights
            .filter_map(|(dir, int)| {
                let dot = dir.dot(&n);
                if dot > 0.0 {
                    Some(dot * int)
                } else {
                    None
                }
            })
            .sum();
        let res = self.color[tr.wli] * total_light;
        //println!("matte: {res}");
        res
    }
}

struct MirrorMaterial {
    shininess: IntensityValue,
    diffusivity: IntensityValue,
    absorbtion: IntensityValue,
}

impl Material for MirrorMaterial {
    fn hit(
        &self,
        tri: &Triangle,
        hit_pos: &Vec3,
        tr: &TracedRay,
        in_lights: &mut dyn Iterator<Item = (Vec3, IntensityValue)>,
        scene: &dyn Tracer,
    ) -> IntensityValue {
        let n = tri.normal_at_pos(hit_pos);
        let total_light: IntensityValue = in_lights
            .filter_map(|(dir, int)| {
                let dot = dir.dot(&n);
                println!("some light");
                if dot > 0.0 {
                    println!("correct light");
                    let h = (dir - tr.ray.dir).normalized();
                    Some(int * n.dot(&h).powf(self.shininess))
                } else {
                    None
                }
            })
            .sum();
        let secondary_intensity = if tr.ttl > 0 {
            let new_dir = tr.ray.dir - 2.0 * tr.ray.dir.dot(&n) * n;// - tr.ray.dir;
            scene.trace(&TracedRay {
                ray: Ray {
                    origin: *hit_pos + EPS * new_dir,
                    dir: new_dir,
                },
                ttl: tr.ttl - 1,
                wli: tr.wli,
            })
        } else {
            0.0
        };
        dbg!(total_light);
        dbg!(secondary_intensity);
        let res =
        self.diffusivity * total_light + (1.0 - self.absorbtion) * secondary_intensity;
        println!("mirror {res}");
        res
    }
}

struct DiffractiveMaterial {
    shininess: IntensityValue,
    diffusivity: IntensityValue,
    absorbtion: IntensityValue,
    d_rad: PositionValue,
    d_perp: PositionValue,
}

impl Material for DiffractiveMaterial {
    fn hit(
        &self,
        tri: &Triangle,
        hit_pos: &Vec3,
        tr: &TracedRay,
        in_lights: &mut dyn Iterator<Item = (Vec3, IntensityValue)>,
        scene: &dyn Tracer,
    ) -> IntensityValue {
        let n = tri.normal_at_pos(hit_pos);
        let b1 = (*hit_pos - tri.struct_center).normalized() * self.d_rad;
        let b2 = n.cross(&(*hit_pos - tri.struct_center)).normalized() * self.d_perp;
        let total_light: IntensityValue = in_lights
            .filter_map(|(dir, int)| {
                let dot = dir.dot(&n);
                //println!("some light");
                if dot > 0.0 {
                    //println!("correct light");
                    Some(11100.0 * int * diffraction_intensity(b1, b2, dir, -tr.ray.dir, index_to_wl(tr.wli), 100.0e-6))
                    /*
                    let a_cos1 = dir.dot(&n);
                    let a_cos2_2 = (1.0 - (1.0 - cos1 * cos1).sqrt() + index_to_wl(tr.wli) / self.d).pow(2.0);
                    let h = (dir - tr.ray.dir).normalized();
                    Some(int * n.dot(&h).powf(self.shininess))
                    */
                } else {
                    None
                }
            })
            .sum();
        let secondary_intensity = if tr.ttl > 0 {
            let new_dir = tr.ray.dir - 2.0 * tr.ray.dir.dot(&n) * n;// - tr.ray.dir;
            scene.trace(&TracedRay {
                ray: Ray {
                    origin: *hit_pos + EPS * new_dir,
                    dir: new_dir,
                },
                ttl: tr.ttl - 1,
                wli: tr.wli,
            })
        } else {
            0.0
        };
        //dbg!(total_light);
        //dbg!(secondary_intensity);
        let res =
        self.diffusivity * total_light + (1.0 - self.absorbtion) * secondary_intensity;
        //println!("mirror {res}");
        res
    }
}

trait Tracer {
    fn trace(&self, tr: &TracedRay) -> IntensityValue;
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
    triangles: Vec<Triangle<'a>>,
    lights: Vec<Box<dyn Light>>,
}

impl<'a> Scene<'a> {
    fn is_obstructed(&self, scr: &ShadowCheckRay) -> bool {
        self.triangles.iter().any(|t| match t.intersect(&scr.ray) {
            Some(t) => t <= scr.light_t,
            None => false,
        })
    }
}

struct RTTracer<'a> {
    scene: &'a Scene<'a>
}

impl Tracer for RTTracer<'_> {
    fn trace(&self, tr: &TracedRay) -> IntensityValue {
        let mut closest_t = f32::INFINITY;
        let mut closest = None;
        for tri in &self.scene.triangles {
            match tri.intersect(&tr.ray) {
                Some(t) => {
                    if t < closest_t {
                        closest_t = t;
                        closest = Some(tri)
                    }
                }
                None => {}
            }
        }

        match closest {
            Some(tri) => {

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
                let res =
                tri.material
                    .hit(tri, &hit_pos, tr, &mut in_lights, self);
                //println!("mat hit {res}");
                res
            }
            None => {
                //println!("miss");
                0.0
            }
        }
    }
}

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
        let rescolor = (0..3)
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
            .unwrap();
        rescolor
    }
}

trait Camera {
    fn render(&self, scene: &dyn Tracer, receptor: &dyn LightReceptor) -> RawImage;
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
    fn render(&self, scene: &dyn Tracer, receptor: &dyn LightReceptor) -> RawImage {
        let mut pixels = vec![ [ 0.0, 0.0, 0.0 ]; (self.pixel_w * self.pixel_h) as usize ];//selVec::new();
        //pixels.reserve_exact((self.pixel_w * self.pixel_h) as usize);

        for y in 0..self.pixel_h {
            for x in 0..self.pixel_w {
                let coeff_u = ((x as PositionValue + 0.5) / (self.pixel_w as PositionValue) - 0.5)
                    * self.proj_w;
                let coeff_v = ((y as PositionValue + 0.5) / (self.pixel_h as PositionValue) - 0.5)
                    * self.proj_h;
                let ray = Ray {
                    origin: self.pos,
                    dir: (self.view_dir + coeff_u * self.proj_u + coeff_v * self.proj_v).normalized(),
                };
                let spectrum = (0..WLS_COUNT)
                    .map(|wli| scene.trace(&TracedRay { ray, ttl: 1, wli }))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                pixels[(y * self.pixel_h + x) as usize] = receptor.process(&spectrum);
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
        let bytes = image.pixels.iter().flat_map(|&rgb| rgb.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect::<Vec<_>>() ).collect::<Vec<_>>();
        image::save_buffer(filename, bytes.as_slice(), image.width, image.height, image::ColorType::Rgb8).unwrap();
    }
}

use std::fs::read_to_string;

fn load_light_receptor(filename: &str) -> RTLightReceptor {
    let rec_values = read_to_string(filename).unwrap().lines()
        .map(|s| {
            let parts = s.split(',').collect::<Vec<_>>();
            (parts[0].parse::<i32>().unwrap(), (0..3).map(|i| parts[i + 1].parse::<IntensityValue>().unwrap()).collect::<Vec<_>>().try_into().unwrap())
        }).collect::<HashMap<i32, [IntensityValue; 3]>>();
    RTLightReceptor {
        receptors: (0..3).map(|i| {
            (0..WLS_COUNT).map(|wli| {
                let wlnm = (index_to_wl(wli) * 1e9).round() as i32;
                rec_values.get(&wlnm).unwrap()[i]
            }).collect::<Vec<_>>().try_into().unwrap()
        }).collect::<Vec<_>>().try_into().unwrap(),
        factor: 10.0,
    }
}

fn construct_materials() -> Vec<Box<dyn Material>> {
    let reddish: Spectrum = (0..WLS_COUNT).map(|wli| if wli > WLS_COUNT * 4 / 5 { 1.0 } else { 0.0 }).collect::<Vec<_>>().try_into().unwrap();
    let reddish_mat = Box::new(MatteMaterial {
        color: reddish
    });
    let mirror_mat = Box::new(MirrorMaterial {
        shininess: 6.0,
        diffusivity: 0.01,
        absorbtion: 0.2
    });
    let cd_mat = Box::new(DiffractiveMaterial {
        shininess: 6.0,
        diffusivity: 0.01,
        absorbtion: 0.2,
        d_rad: 1.5e-6,
        d_perp: 3.0e-6,
    });
    vec![ reddish_mat, mirror_mat, cd_mat ]
}

fn construct_scene0<'a>(materials: &'a Vec<Box<dyn Material>>) -> Scene<'a> {
    let red_tri = Triangle {
        verts: [ (Vec3::new(0.0, -0.5, 1.0), Vec3::new(0.0, 1.0, -1.0).normalized()),
        (Vec3::new(-0.5, 0.5, 2.0), Vec3::new(0.0, 1.0, -1.0).normalized()),
        (Vec3::new(0.5, 0.5, 2.0), Vec3::new(0.0, 1.0, -1.0).normalized()) ],
        struct_center: Vec3::zero(),
        material: &*materials[0]//&*reddish_mat
    };
    let mirror_tri = Triangle {
        verts: [ (Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, -1.0).normalized()),
        (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, -1.0).normalized()),
        (Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 1.0, -1.0).normalized()) ],
        struct_center: Vec3::zero(),
        material: &*materials[1]//mirror_mat
    };
    let main_light = DirectionalLight {
        dir: Vec3::new(0.3, -0.2, 4.0).normalized(),
        intensity: 1.0,
    };
    Scene {
        triangles: vec![ red_tri, mirror_tri ],
        lights: vec![ Box::new(main_light) ]
    }
}

fn construct_scene1<'a>(materials: &'a Vec<Box<dyn Material>>) -> Scene<'a> {
    let red_tri = Triangle {
        verts: [ (Vec3::new(0.0, 0.5, 1.0), Vec3::new(1.0, 0.0, -1.0).normalized()),
        (Vec3::new(-1.0, 0.5, 0.0), Vec3::new(1.0, 0.0, -1.0).normalized()),
        (Vec3::new(-0.5, -0.5, 0.5), Vec3::new(1.0, 0.0, -1.0).normalized()) ],
        struct_center: Vec3::zero(),
        material: &*materials[0]//&*reddish_mat
    };
    let mirror_tri0 = Triangle {
        verts: [ (Vec3::new(0.0, 2.0, 1.0), Vec3::new(-1.0, 0.0, -1.0).normalized()),
        (Vec3::new(0.0, -2.0, 1.0), Vec3::new(-1.0, 0.0, -1.0).normalized()),
        (Vec3::new(2.0, 2.0, -1.0), Vec3::new(-1.0, 0.0, -1.0).normalized()) ],
        struct_center: Vec3::new(1.0, 0.0, 0.0),
        material: &*materials[2]//mirror_mat
    };
    let mirror_tri1 = Triangle {
        verts: [ (Vec3::new(0.0, -2.0, 1.0), Vec3::new(-1.0, 0.0, -1.0).normalized()),
        (Vec3::new(2.0, 2.0, -1.0), Vec3::new(-1.0, 0.0, -1.0).normalized()),
        (Vec3::new(2.0, -2.0, -1.0), Vec3::new(-1.0, 0.0, -1.0).normalized()) ],
        struct_center: Vec3::new(1.0, 0.0, 0.0),
        material: &*materials[2]//mirror_mat
    };
    let main_light = DirectionalLight {
        dir: Vec3::new(0.0, -0.0, 4.0).normalized(),
        intensity: 1.0,
    };
    Scene {
        triangles: vec![ red_tri, mirror_tri0, mirror_tri1 ],
        lights: vec![ Box::new(main_light) ]
    }
}

fn construct_tracer<'a>(scene: &'a Scene) -> Box::<dyn Tracer + 'a> {
    Box::new(RTTracer {
        scene
    })
}

fn construct_camera() -> RTCamera {
    RTCamera {
        pos: Vec3::new(0.0, 0.0, -6.0),
        view_dir: Vec3::new(0.0, 0.0, 1.0),
        proj_u: Vec3::new(1.0, 0.0, 0.0),
        proj_v: Vec3::new(0.0, -1.0, 0.0),
        proj_w: 0.4,
        proj_h: 0.4,
        pixel_w: 128,
        pixel_h: 128
    }
}

fn main() {
    let receptor = load_light_receptor("receptor.csv");
    let materials = construct_materials();
    let scene = construct_scene1(&materials);
    let tracer = construct_tracer(&scene);
    let camera = construct_camera();
    let image = camera.render(tracer.as_ref(), &receptor);
    RTImageWriter::write_image(image, "out.png");
}

// TODO: make performance improvements to render a larger image e.g.
// -- don't render all wavelengths when the ray doesn't hit anything
// -- add fast-paths for rays not hitting objects
// -- compile in release
