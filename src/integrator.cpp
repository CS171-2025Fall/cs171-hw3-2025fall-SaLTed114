#include "rdr/integrator.h"

#include <omp.h>

#include "rdr/bsdf.h"
#include "rdr/camera.h"
#include "rdr/canary.h"
#include "rdr/film.h"
#include "rdr/halton.h"
#include "rdr/interaction.h"
#include "rdr/light.h"
#include "rdr/math_aliases.h"
#include "rdr/math_utils.h"
#include "rdr/platform.h"
#include "rdr/properties.h"
#include "rdr/ray.h"
#include "rdr/scene.h"
#include "rdr/sdtree.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * Intersection Test Integrator's Implementation
 *
 * ===================================================================== */

void IntersectionTestIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {
        // TODO(HW3): generate #spp rays for each pixel and use Monte Carlo
        // integration to compute radiance.
        //
        // Useful Functions:
        //
        // @see Sampler::getPixelSample for getting the current pixel sample
        // as Vec2f.
        //
        // @see Camera::generateDifferentialRay for generating rays given
        // pixel sample positions as 2 floats.

        // You should assign the following two variables
        // const Vec2f &pixel_sample = ...
        // auto ray = ...

        // After you assign pixel_sample and ray, you can uncomment the
        // following lines to accumulate the radiance to the film.
        //
        //
        // Accumulate radiance
        // assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        // assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        // const Vec3f &L = Li(scene, ray, sampler);
        // camera->getFilm()->commitSample(pixel_sample, L);

        const Vec2f &pixel_sample = sampler.getPixelSample();
        auto ray = camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);

        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        const Vec3f &L = Li(scene, ray, sampler);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f IntersectionTestIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  Vec3f color(0.0);

  // Cast a ray until we hit a non-specular surface or miss
  // Record whether we have found a diffuse surface
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  for (int i = 0; i < max_depth; ++i) {
    interaction      = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    // Perform RTTI to determine the type of the surface
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // Set the outgoing direction
    interaction.wo = -ray.direction;

    if (!intersected) {
      break;
    }

    if (is_perfect_refraction) {
      // We should follow the specular direction
      // TODO(HW3): call the interaction.bsdf->sample to get the new direction
      // and update the ray accordingly.
      //
      // Useful Functions:
      // @see BSDF::sample
      // @see SurfaceInteraction::spawnRay
      //
      // You should update ray = ... with the spawned ray
      // UNIMPLEMENTED;

      Float pdf;
      interaction.bsdf->sample(interaction, sampler, &pdf);

      ray = interaction.spawnRay(interaction.wi);
      continue;
    }

    if (is_ideal_diffuse) {
      // We only consider diffuse surfaces for direct lighting
      diffuse_found = true;
      break;
    }

    // We simply omit any other types of surfaces
    break;
  }

  if (!diffuse_found) {
    return color;
  }

  color = directLighting(scene, interaction);
  return color;
}

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction) const {
#define USE_ADVANCED_LIGHTS
#define ENABLE_AREA_LIGHT

#ifdef USE_ADVANCED_LIGHTS
  Vec3f color(0, 0, 0);

  const BSDF* bsdf = interaction.bsdf;
  bool is_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;
  if (!bsdf || !is_diffuse) return color;

  static const std::vector<Vec3f> light_positions = {
    Vec3f(0.0f, 1.5f, 0.5f),   // main light
    Vec3f(-0.8f, 1.4f, -0.5f), // left warm fill light
    Vec3f(0.9f, 1.6f, 0.3f)    // right cool fill light
  };

  static const std::vector<Vec3f> light_intensities = {
    Vec3f(17.0f, 12.0f, 5.0f), // main light
    Vec3f(4.0f, 3.0f, 2.0f),   // left warm fill light
    Vec3f(3.0f, 4.5f, 6.0f)    // right cool fill light
    // Vec3f(0.0f, 0.0f, 0.0f),
    // Vec3f(0.0f, 0.0f, 0.0f),
    // Vec3f(0.0f, 0.0f, 0.0f)
  };

  for (size_t i = 0; i < light_positions.size(); ++i) {
    Vec3f light_pos = light_positions[i];
    Vec3f light_intensity = light_intensities[i];

    Float dist_to_light = Norm(light_pos - interaction.p);
    Vec3f wi = Normalize(light_pos - interaction.p);

    DifferentialRay shadow_ray(interaction.p, wi);
    SurfaceInteraction occ;
    bool blocked = scene->intersect(shadow_ray, occ);

    if (blocked) {
      Float hit_dist = Norm(occ.p - interaction.p);
      if (hit_dist <  dist_to_light) {
        // occluded
        continue; 
      }
    }

    Float cos_theta = std::max(Dot(wi, interaction.normal), 0.0f);
    if (cos_theta <= 0.0f) continue;

    Vec3f brdf = bsdf->evaluate(interaction);

    color += brdf * cos_theta * light_intensity / (dist_to_light * dist_to_light);
  }

  color *= 0.25f;
#ifdef ENABLE_AREA_LIGHT
  Vec3f rect_center(0.0f, 1.4f, 0.0f);
  float width  = 1.2f;
  float height = 1.2f;

  Vec3f ex(width, 0.0f, 0.0f);
  Vec3f ey(0.0f, 0.0f, height);
  
  Vec3f area_light_radiance(10.0f, 10.0f, 10.0f);

  Vec3f area_accum(0.0f, 0.0f, 0.0f);

  for (int s = 0; s < 16/*AREA_LIGHT_SAMPLES*/; s++) {
    float u = float(rand()) / RAND_MAX;
    float v = float(rand()) / RAND_MAX;

    Vec3f light_pos = rect_center + (u - 0.5f) * ex + (v - 0.5f) * ey;

    Float dist_to_light = Norm(light_pos - interaction.p);
    Vec3f wi = Normalize(light_pos - interaction.p);

    DifferentialRay shadow_ray(interaction.p, wi);
    SurfaceInteraction occ;
    bool blocked = scene->intersect(shadow_ray, occ);

    if (blocked && Norm(occ.p - interaction.p) < dist_to_light) {
      continue;
    }

    float cos_theta = std::max(Dot(wi, interaction.normal), 0.0f);
    if (cos_theta <= 0.0f) continue;

    Vec3f brdf = bsdf->evaluate(interaction);

    area_accum += brdf * cos_theta * area_light_radiance / (dist_to_light * dist_to_light);
  }

  area_accum /= 16.0f/*AREA_LIGHT_SAMPLES*/;
  color += area_accum;

#endif

  return color * 0.3f;
#else
  Vec3f color(0, 0, 0);
  Float dist_to_light = Norm(point_light_position - interaction.p);
  Vec3f light_dir     = Normalize(point_light_position - interaction.p);
  auto test_ray       = DifferentialRay(interaction.p, light_dir);

  // TODO(HW3): Test for occlusion
  //
  // You should test if there is any intersection between interaction.p and
  // point_light_position using scene->intersect. If so, return an occluded
  // color. (or Vec3f color(0, 0, 0) to be specific)
  //
  // You may find the following variables useful:
  //
  // @see bool Scene::intersect(const Ray &ray, SurfaceInteraction &interaction)
  //    This function tests whether the ray intersects with any geometry in the
  //    scene. And if so, it returns true and fills the interaction with the
  //    intersection information.
  //
  //    You can use iteraction.p to get the intersection position.
  //
  // UNIMPLEMENTED;

  SurfaceInteraction shadow_interaction;
  bool occluded = scene->intersect(test_ray, shadow_interaction);

  if (occluded) {
    Float dist_to_intersection = Norm(shadow_interaction.p - interaction.p);
    if (dist_to_intersection < dist_to_light) {
      return color;  // Occluded
    }
  }

  // Not occluded, compute the contribution using perfect diffuse diffuse model
  // Perform a quick and dirty check to determine whether the BSDF is ideal
  // diffuse by RTTI
  const BSDF *bsdf      = interaction.bsdf;
  bool is_ideal_diffuse = dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

  if (bsdf != nullptr && is_ideal_diffuse) {
    // TODO(HW3): Compute the contribution
    //
    // You can use bsdf->evaluate(interaction) * cos_theta to approximate the
    // albedo. In this homework, we do not need to consider a
    // radiometry-accurate model, so a simple phong-shading-like model is can be
    // used to determine the value of color.

    // The angle between light direction and surface normal
    Float cos_theta =
        std::max(Dot(light_dir, interaction.normal), 0.0f);  // one-sided

    // You should assign the value to color
    // color = ...
    // UNIMPLEMENTED;

    if (bsdf != nullptr && is_ideal_diffuse) {
      Vec3f brdf = bsdf->evaluate(interaction);
      
      // color = albedo * cos_theta * light_intensity / distance^2
      color = 0.3f * brdf * cos_theta * point_light_flux / (dist_to_light * dist_to_light);
    }
  }

  return color;
#endif
}

/* ===================================================================== *
 *
 * Path Integrator's Implementation
 *
 * ===================================================================== */

void PathIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::Li(
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

/* ===================================================================== *
 *
 * New Integrator's Implementation
 *
 * ===================================================================== */

// Instantiate template
// clang-format off
template Vec3f
IncrementalPathIntegrator::Li<Path>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
template Vec3f
IncrementalPathIntegrator::Li<PathImmediate>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
// clang-format on

// This is exactly a way to separate dec and def
template <typename PathType>
Vec3f IncrementalPathIntegrator::Li(  // NOLINT
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

RDR_NAMESPACE_END
