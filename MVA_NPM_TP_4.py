
############## Import Dependencies ################

from PIL import Image
import math
import numpy as np



##################################################################################
############################### Helper classes ###################################
##################################################################################



# We create the material class
class Material:
  def __init__(self, albedo_rgb, diffuse_coef, specular_coef=None, shininess=None):
    """
      Class to be used to create materials
      args:
      albedo_rgb : (array) 3-length list of rdg colors to apply 
      diffuse_coef : (int) material diffuse coefficient
      specular_coef : (int) Specular light reflection coef (optional, regarding the shader)
      shininess : (int) Shininess coefficient
    """
    self.albedo = albedo_rgb
    self.diffuse_coef = diffuse_coef

    self.specular_coef = specular_coef
    self.shininess = shininess

# Lightsource class
class LightSource:
  """
    Class to be used to create light source
    This implements a directional light source
    args:
      direction : (3-length list of int) the light direction
      color : rgb array
      intensity : int
  """
  def __init__(self, direction, color, intensity):
    self.direction = np.array(direction)
    self.rgb = np.array(color)
    self.intensity = intensity



#################################################################################
################################ Diffuse Shader #################################
#################################################################################


def shade(normalImage, material, light_source):
  """
  This function applies a diffuse shader to a normal image
  args:
    normalImage : PIL normals image
    material : the material to apply to the normal map
    light_source :

  returns:
    PIL image with applied shade

  """
  n = np.array(normalImage)[:,:,:3] # we get the normals 
  n_copy = n.copy() # save a copy for thresholding the dark zone later
  
  n_norm = np.linalg.norm(n, axis=-1, keepdims=True) # Normalize the normals
  n_norm[n_norm==0] = 1
  n = n/n_norm
  light_dir =   light_source.direction
  light_dir = light_dir/np.linalg.norm(light_dir) # normalize the light direction

  # Lambert BRDF
  f_d = material.diffuse_coef * material.albedo /np.pi

  # rendering equation
  out = (light_source.intensity )*f_d * np.maximum(np.expand_dims((n*light_dir).sum(-1), axis=-1), 0)
  out[out>255]=255
  out[n_copy<5] = 0
  im = Image.fromarray(out.astype(np.uint8))
  return im



def try_different_diffuse_coef(normal_img):
  light = LightSource([1,400,-125],[255.,255.,255.], 1.413) # ideal intensity 10  ### the one I am using
  imgs = []
  new_im = Image.new('RGB', (500,293))
  for pos ,coef in zip([(0,0), (250,0), (0,146), (250, 146)], [1.,5.,7., 15.]):
    material = Material(np.array([0,191,255]), coef)
    img = shade(normal_img, material, light) 
    img.thumbnail((250,250), Image.ANTIALIAS)
    new_im.paste(img, pos)

  return new_im








#################################################################################
################################ Specular Shader #################################
#################################################################################


def specular_shade(normalImage, material, light_source):

  out_dir = [1,1,1]

  n = np.array(normalImage)[:,:,:3]
  n_copy = n.copy()
  
  n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
  n_norm[n_norm==0] = 1
  n = n/n_norm
  light_dir =   light_source.direction
  light_dir = light_dir/np.linalg.norm(light_dir)

  f_d = material.diffuse_coef * material.albedo /np.pi
  w_h = (light_dir +  out_dir)/(2. * np.linalg.norm(light_dir +  out_dir))
  f_s = material.specular_coef * np.power(np.expand_dims((n*w_h).sum(-1), axis=-1), material.shininess)
  
  out = (light_source.intensity )*(f_d + f_s) * np.maximum(np.expand_dims((n*light_dir).sum(-1), axis=-1), 0)
  out[out>255] = 255
  out[n_copy<5] = 0
  im = Image.fromarray(out.astype(np.uint8))
  return im


def try_different_spec_shininess(normal_image):
  light = LightSource([1,400,-125],[255.,255.,255.], 1.413) 

  new_im = Image.new('RGB', (750,440))

  for pos ,(spec, shininess) in zip([(0,0), (250,0), (500, 0),
                        (0,146), (250,146), (500,146),
                        (0,293), (250, 293), (500, 293)], 
                       
                       [(100, 0.), (100, .5), (100,1.),
                        (255, 0.), (255, .5), (255, 1.),
                        (600, 0.), (600, .5), (600, 1.)]):
    material = Material(np.array([62,28,82]), 4.5,spec, shininess)
    img = specular_shade(normal_image, material, light) 
    img.thumbnail((250,250), Image.ANTIALIAS)
    new_im.paste(img, pos)

  return new_im




#################################################################################
############################# Microfacet Shader #################################
#################################################################################


def D(alpha, n, w_h):
  prod = np.expand_dims((n*w_h).sum(-1), axis=-1)
  coef = alpha**2 - 1
  return alpha**2/(np.pi*(1 + prod**2 * coef)**2)

def G1(alpha, n, w):
  prod = np.expand_dims((n*w).sum(-1), axis=-1)
  k = alpha * np.sqrt(2/np.pi)
  return prod /(prod * (1-k) + k)

def G(alpha, w_i, w_o, n):
  return G1(alpha, n, w_i)*G1(alpha, n, w_o)

def F(n, w_h,f_0):
  prod = np.expand_dims((n*w_h).sum(-1), axis=-1)
  return f_0 + (1-f_0)*np.power(2, (-5.55473*prod - 6.98316)*prod)

def micro_facette_shader(normalImage, material,  alpha, f_0,   light_source):

  out_dir = [1,1,1]

  n = np.array(normalImage)[:,:,:3]
  n_copy = n.copy()
  
  n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
  n_norm[n_norm==0] = 1
  n = n/n_norm
  light_dir =   light_source.direction
  light_dir = light_dir/np.linalg.norm(light_dir)

  f_d = material.diffuse_coef * material.albedo /np.pi

  w_h = (light_dir +  out_dir)/( np.linalg.norm(light_dir +  out_dir))

  prod_i = np.expand_dims((n*light_dir).sum(-1), axis=-1)
  prod_o = np.expand_dims((n*out_dir).sum(-1), axis=-1)
  prod_i[prod_i==0]=1
  prod_o[prod_o==0]=1
  f_s = (D(alpha, n, w_h)*F(n, w_h,f_0)*G(alpha, light_dir, out_dir,n))/(4*prod_i*prod_o)
  
  out = (light_source.intensity )*(f_d + f_s) * np.maximum(np.expand_dims((n*light_dir).sum(-1), axis=-1), 0)
  out[n_copy<5] = 0
  out[out>255] = 255
  im = Image.fromarray(out.astype(np.uint8))
  return im


def try_different_spec_rough(normal_image):
  light = LightSource([1,400,-125],[255.,255.,255.], 1.413) 

  new_im = Image.new('RGB', (750,440))

  for pos ,(alpha, f_0) in zip([(0,0), (250,0), (500, 0),
                        (0,146), (250,146), (500,146),
                        (0,293), (250, 293), (500, 293)], 
                       
                       [(2.9, 100.), (2.9, 500), (2.9,1000),
                        (2.95, 100.), (2.95, 500), (2.95, 1000),
                        (2.965, 100.), (2.965, 500), (2.965, 1000)]):
    material = Material(np.array([125,10,10]), 4.5)
    img = micro_facette_shader(normal_image, material,alpha, f_0, light) 
    img.thumbnail((250,250), Image.ANTIALIAS)
    new_im.paste(img, pos)

  return new_im




if __name__ == "__main__":

  #################### Load Image ######################
  img = Image.open("normal.png")




  #######################################################
  ############# Test diffuse shading ####################
  #######################################################

  # First create light source and material (lightsource intensity = 1.413 - sunlight intensity on earth
  #and diffuse coefficient = 4.5 - gold diffuse coefficient under certain conditions)
  light = LightSource([1,400,-125],[255.,255.,255.], 1.413) # white light
  material = Material(np.array([0,191,255]), 4.5) # Cyan color

  # use the shader and save image
  print("Saving Diffuse shading result image")
  im = shade(img, material, light) 
  im.save("diffuse_shading.png")

  # Now we are going to try different diffuse coefficients and see how the image varies
  # The higher the intensity, the brighter the color
  # More details in the report
  im = try_different_diffuse_coef(img)
  print("Saving Diffuse shading (with different diffuse coefficients) result image")
  im.save("different_diffuse_coefficients.png")





  ##########################################################
  ########## Test Blinn-Phong Specular material ############
  ##########################################################

  light = LightSource([1,400,-125],[255.,255.,255.], 1.413) # ideal intensity 10  ### the one I am using
  material = Material(np.array([0,191,255]), 4.5, 255., .7)

  im = specular_shade(img, material, light )
  print("Saving Diffuse + Blinn-Phong shading result image")
  im.save("Blinn-Phong_shading.png")

  # We tried different specular and shininess coefficients
  # More details in the report
  im = try_different_spec_shininess(img)
  print("Saving Diffuse + Blinn-Phong shading (with different specular and shininess coefficients) result image")
  im.save("Blinn-Phong_different_coefficients.png")




  ###########################################################
  ################## Test Microfacet model ##################
  ###########################################################

  # We create the light source and different materials with different colors
  light = LightSource([1,400,-125],[255.,255.,255.], 1.413) # ideal intensity 10  ### the one I am using
  materials = [
    Material(np.array([0,191,255]), 2), # Cyan (dark cyan)
    Material(np.array([62,28,82]), 4.5), # purple
    Material(np.array([125,10,10]), 4.5), # red
    Material(np.array([70,150,40]), 2) # green
  ]
  colors = ["cyan", "purple", "red", "green"]
  print("Saving Diffuse + Microfacet shading result image")
  for color, material in zip(colors, materials):
    im = micro_facette_shader(img, material, 2.965, 70., light)
    im.save("{}_microfacet_shading.png".format(color))


  # We tried different specular and roughness coeffcients
  # More details in the report
  print("Saving Diffuse + Microfaset (tested wth different coefficients) result image")
  im = try_different_spec_rough(img)
  im.save("Microfacet_different_coefficients.png")


  print("\n\n\nPlease have a look at your folder to visualize the results.")


