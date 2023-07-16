#entries by the command line
from copy import deepcopy #used on mean and median 
import argparse ; import os #commands and files
from math import hypot #calculating the sobel

parse=argparse.ArgumentParser()
parse.add_argument("--imgpath",type=str,help="image path")
parse.add_argument("--op",type=str,choices=["sobel","thresholding","sgt","mean","median"],help="operations to the image")
parse.add_argument("--t",type=int,help="argument for the thresholding", default=127)
parse.add_argument("--dt",type=float,help="argument for sgt",default=1)
parse.add_argument("--k",type=int,help="argument for mean/median",default=3)
parse.add_argument("--outputpath",type=str,help="path where the image will be saved, you shouldn't include the name of the file",
                   default=os.getcwd())
args=parse.parse_args()

#storing variables 
img_path=args.imgpath
operation=args.op
output_path=args.outputpath
if operation=="thresholding": t=args.t
if operation=="sgt" : dt=args.dt
if operation in ["mean","median"] : k=args.k

##importing image
def open_image(img_path : str) -> list:
  '''
  Creates an Image object using the img_path
  '''
  with open(img_path,"r") as file:
    lines=file.readlines()
    lines=[line for line in lines if line.count("#")==0] #remove comments
    img_type=lines[0] #The code only works with P2 images
    first_line=lines[1].split()  
    sizes={"width":int(first_line[0]), "height":int(first_line[1])} #to access the list is on reverse order
    max_val=lines[2]
    pixels=[]
    for line in lines[3:]: 
      row = [int(x) for x in line.rstrip().split()]
      pixels.append(row)
  return Image(pixels=pixels,img_type=img_type,max_val=max_val,sizes=sizes)

class Image:
  def __init__(self,pixels:list,sizes:dict,max_val:int,img_type:str):
    self.pixels=pixels
    self.sizes=sizes
    self.max_val=max_val
    self.img_type=img_type
    def create_histogram(self,pixels : list) -> list:
      frequencies={j : 0 for j in range(256)}
      for line in range(self.sizes["height"]):
        for column in range(self.sizes["width"]):
          frequencies[pixels[line][column]]+=1
      return frequencies
    self.histogram=create_histogram(self,pixels)

  def thresholding(self, T=127,g_returner=False) -> list:
    '''
    If g_returner=False, it creates a new Image object with thresholding applied
    If g_returner=True, it returnes defines the G1 and G2 lists which are the pixels
    bigger than T1 and smaller than T2 
    '''
    thresholding_image = [[0] * self.sizes['width'] for _ in range(self.sizes['height'])]
    if g_returner:
      G1=[] ; G2=[]
    for line in range(self.sizes['height']):
      for number in range(self.sizes['width']):
        if self.pixels[line][number]>T:
          thresholding_image[line][number]=255
          if g_returner:
            G1.append(self.pixels[line][number])
        else:
          thresholding_image[line][number]=0
          if g_returner:
            G2.append(self.pixels[line][number])
    if g_returner:
        return G1,G2
    return Image(pixels=thresholding_image,max_val=self.max_val,sizes=self.sizes,img_type=self.img_type)
  
  def sgt(self,dt=1,T_returner=False):
    '''
    this function calls the thresholding method of the class
    and applies it until the condition abs(t_new - t_old)<=dt is true
    if T_returner, it returns the value of T
    '''
    thresholding=self.thresholding
    T_initial=127
    T=T_initial #starting the loop
    while True:
      G1,G2=thresholding(T,g_returner=True)
      T_new=int(0.5*(sum(G1)/len(G1) + sum(G2)/len(G2)))
      if abs(T_new-T)<=dt:
        break
      T=T_new
    if T_returner:
      return T_new
    filtered_image=self.thresholding(T_new)
    return Image(filtered_image.pixels,filtered_image.sizes,filtered_image.max_val,filtered_image.img_type)
  def sobel(self):
    empty_list=[[0] * self.sizes["width"] for _ in range(self.sizes["height"])]
    #partial images
    Gx_img=deepcopy(empty_list)
    Gy_img=deepcopy(empty_list)
    #kernels
    Gx=[[1,0,-1],[2,0-2],[1,0,-1]]
    Gy=[[1,2,1],[0,0,0],[-1,-2,-1]]
    #resulting img
    filtered_image=deepcopy(empty_list)
    #to generate both imgs, Gx_img and Gy_img
    # the loop is applied two times,
    # Gx_img -> times=0, Gy_times -> times=1
    for times in range(2):
      for line in range(self.sizes["height"]):
          for column in range(self.sizes["width"]):
            cumulative_sum=0
            for dx in range(-1,1+1): 
              for dy in range(-1,1+1):
                try:
                  if times==0:
                    cumulative_sum+=(self.pixels[line+dx][column+dy]*Gx[1+dx][1+dy])
                  if times==1:
                    cumulative_sum+=(self.pixels[line+dx][column+dy]*Gy[1+dx][1+dy])
                except:... #considering zero
            if times==0: Gx_img[line][column]=cumulative_sum
            if times==1: Gy_img[line][column]=cumulative_sum
    bigger_value=0
    for line in range(self.sizes["height"]):
      for column in range(self.sizes[ "width"]):
        new_px=hypot(Gx_img[line][column],Gy_img[line][column])
        filtered_image[line][column]=new_px
        #the following line is searching for the bigger pixel inside the img
        if new_px>bigger_value: bigger_value=new_px
    #now the img pixels are normalized to 8bit 
    for line in range(self.sizes["height"]):
      for column in range(self.sizes["width"]):
        filtered_image[line][column]=round(256*filtered_image[line][column]/bigger_value)

    return Image(pixels=filtered_image,sizes=self.sizes,max_val=self.max_val,img_type=self.img_type)

  def mean(self,k=3):
    '''k is the size of kernel matrix (k x k)
      d will be defined as the distance from the center
      up to the edge of the kernel matrix
    '''
    new_pixels=deepcopy(self.pixels)
    d=k//2
    for line in range(self.sizes["height"]):
        for column in range(self.sizes["width"]):
          average=0
          for dx in range(-d,d+1):
            for dy in range(-d,d+1):
              try:
                average+=self.pixels[line+dx][column+dy]
              except:... #consedering zero
          average=average/(k*k)
          new_pixels[line][column]=int(average)
    return Image(pixels=new_pixels,sizes=self.sizes,max_val=self.max_val,img_type=self.img_type)
  
  def median(self,k=3):
    '''k is the size of kernel matrix (k x k)
      d will be defined as the distance from the center
      up to the edge of the kernel matrix
    '''
    new_pixels=deepcopy(self.pixels)
    d=k//2
    for line in range(self.sizes["height"]):
        for column in range(self.sizes["width"]):
          kernel=[]
          for dx in range(-d,d+1):
            for dy in range(-d,d+1):
              try:
                kernel.append(self.pixels[line+dx][column+dy])
              except: 
                kernel.append(0)
          kernel=sorted(kernel)
          middle_index=(k*k)//2 
          new_pixels[line][column]=kernel[middle_index]
    return Image(pixels=new_pixels,sizes=self.sizes,max_val=self.max_val,img_type=self.img_type)

def save(image: Image,output_path:str,file_name:str) -> None:
  '''Saves an pgm image into the desired output'''
  with open(output_path+"//"+file_name,"w") as file:
    file.write(image.img_type+"\n")
    file.write(str(image.sizes["width"])+" "+str(image.sizes["height"])+"\n")
    file.write(image.max_val+"\n")
    lines=[]
    for row in image.pixels:
      string_row = ' '.join([str(item) for item in row])+' \n'
      lines.append(string_row)
    for line in lines:
        file.write(line)

if __name__=="__main__":
  image=open_image(img_path)
  if operation=="thresholding":
    filtered_image=image.thresholding(t)
  if operation=="sgt":
    filtered_image=image.sgt(dt)
  if operation=="mean":
    filtered_image=image.mean(k)
  if operation=="median":
    filtered_image=image.median(k)
  if operation=="sobel":
    filtered_image=image.sobel()
  if operation==None:
    filtered_image=image
  file_name=operation+".pgm"
  save(filtered_image,output_path,file_name)
  ##header image features
  print(f"magic_number {image.img_type}")
  print(f"dimensions({image.sizes['height']}, {image.sizes['width']})")
  print(f"maxval {image.max_val}")
  #numerical image features
  pixels_one_list=[pixel for pixel_line in image.pixels for pixel in pixel_line]
  average=int(sum(pixels_one_list)/len(pixels_one_list))
  median=sorted(pixels_one_list)
  list_size=len(median)
  if list_size%2==1:
    median=median[list_size//2]
  else:
    median=0.5*(median[list_size//2]+median[list_size//2 +1])
    median=int(median)
  print(f"mean ={average}")
  print(f"median ={median}")
  print(f"T ={image.sgt(dt=1,T_returner=True)}")