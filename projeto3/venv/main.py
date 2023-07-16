#################################################
# Projeto 3 - Programação Orientada a Objetos   #
# Prof. Gonzalo Travieso                        #
# IFSC/USP                                      #
# Alunos:                                       #
#   - Breno Henrique Pelegrin da Silva          #
#   - Vinicius Sousa Dutra                      #
# Data: 29/06/2023                              #
# Licença: MIT                                  #
#################################################

import argparse
import sys
import os
import copy
# Disclaimer about the packages usage:
# The "os" package will be used to join paths and filenames and provide compatibility between UNIX-based systems and DOS-based systems.
# The "sys" package will be used to provide clean exit for production environment.
# The "argparse" package will be used to parse CLI arguments.
# The "copy" package will be used to do deepcopy of arrays when applying filters.

def flatten(array):
    return [item for sublist in array for item in sublist]

def mean(array):
    return(sum(array)/len(array))

def median(array):
    lst = sorted(array)
    if len(lst) % 2 == 0:
        return (lst[len(lst) // 2] + lst[(len(lst) - 1) // 2]) / 2
    else:
        return lst[len(lst) // 2]

class Image:
    def __init__(self, fromFile: bool = False, filePath: str = '', magicNumber: str = '', imagePixels: list = []):

        if(fromFile):
            assert filePath != '', "To create an Image from file, pass the filePath argument."

            self._fromFile = True
            self._filePath = filePath

            fileData = self.__readFromPGM(self._filePath)

            self._magicNumber = fileData['type']
            self._imagePixels = fileData['pixels']
            self._maxval = fileData['maxval']

        else:
            assert magicNumber != '', "To manually create an Image, pass the magicNumber argument."
            assert imagePixels != [], "To manually create an Image, pass the imagePixels argument."

            self._fromFile = False
            self._filePath = ''

            self._magicNumber = magicNumber
            self._imagePixels = imagePixels
            self._maxval = self.__getMaxvalFromPixels(self._imagePixels)

        self._width, self._height = self.__getDimensionsFromPixels(self._imagePixels)

        self._meanIntensity = self.__getMeanIntensityFromPixels(self._imagePixels)
        self._medianIntensity = self.__getMedianIntensityFromPixels(self._imagePixels)
        self._imageHistogram = self.__getHistogramFromPixels(self._imagePixels)

        # sgt method must ran for this attribute to be set.
        self._T = None

    def __readFromPGM(self, filePath: str) -> dict:  
        """
            Reads a PGM file and convert its content into usable data.
            Only works for PGM files past July/2000. After that, PGM's could have more than one image in one file.
            
            Can read:
                - All P2 files (ASCII 8-bit)
                - P5 files with 8-bit pixels (maxval<=255), with pixels encoded in the format 'byte byte byte byte...'

            Args:
                filePath (str): path to the pgm file
        """
        data = {
            'type': None,
            'width': None,
            'height': None,
            'maxval': None,
            'pixels': []
        }
        
        linesRaw = []
        with open(filePath, 'rb') as file:
            while line := file.readline():
                linesRaw.append(line)

        comment_index = None
        for i in range(3):
            # Header have AT LEAST 3 lines (magic number, dimensions, maxval).
            # It can have AT MOST 4 lines, of which one CAN BE a comment or blank line.
            # The header is encoded in ASCII.
            decoded = linesRaw[i].decode('ASCII').rstrip()
            if('#' in decoded or decoded == ''):
                comment_index = i

        if(comment_index):
            # If comment or blank line was found, delete it, which makes header have EXACTLY 3 lines.
            del linesRaw[comment_index]
        
        header = [i.decode('ASCII').rstrip() for i in linesRaw[:3]]
        magic = header[0]
        dimensions = header[1].split()
        maxval = header[2]

        data['type'] = magic
        data['width'] = int(dimensions[0])
        data['height'] = int(dimensions[1])
        data['maxval'] = int(maxval)

        if(magic == 'P2'):
            pixelsRaw = [i.decode('ASCII').rstrip() for i in linesRaw[3:]]

            for line in pixelsRaw:
                row = [int(x) for x in line.split()]
                data['pixels'].append(row)

            return data

        elif(magic == 'P5'):
            # P5 files encode pixels in plain bytes.
            # This code only reads P5 files with maxval <= 255 which encodes 1 byte per pixel, providing support for P2 <-> P5 conversion.
            pixelsRaw = linesRaw[3:][0]
            
            # Splits pixelsRaw in chunks of width bytes, resulting in height rows of width bytes.
            pixelsBytes = [pixelsRaw[i:i + data['width']] for i in range(0, len(pixelsRaw), data['width'])]

            for i in range(len(pixelsBytes)):
                # Converts each byte into int
                row = [int(v) for v in pixelsBytes[i]]
                data['pixels'].append(row)
            return data

        else:
            raise Exception(f"File '{filePath}' has invalid magic number (neither P5 or P2).")
    
    def to_P5(self):
        newImage = Image(
            fromFile = False,
            magicNumber = 'P5',
            imagePixels = copy.deepcopy(self._imagePixels))
        return newImage
    
    def to_P2(self):
        newImage = Image(
            fromFile = False,
            magicNumber = 'P2',
            imagePixels = copy.deepcopy(self._imagePixels))
        return newImage

    def saveToPGM(self, outputPath: str):
        """
            Writes a PGM file, converting Image pixels to PGM.
            Can output in both P2 and P5 formats.

            Args:
                outputPath (str): path to the output pgm file.
        """
        if(self._magicNumber == 'P2'):
            lines = []
            lines.append(f'{self._magicNumber}\n')
            lines.append(f'{self._width} {self._height}\n')
            lines.append(f'{self._maxval}\n')
            
            for row in self._imagePixels:
                stringRow = ' '.join([str(item) for item in row])+' \n'
                lines.append(stringRow)
            
            with open(outputPath, 'w') as outputFile:
                for line in lines:
                    outputFile.write(line)

        elif(self._magicNumber == 'P5'):
            lines = []
            lines.append(f'{self._magicNumber}\n'.encode('ASCII'))
            lines.append(f'{self._width} {self._height}\n'.encode('ASCII'))
            lines.append(f'{self._maxval}\n'.encode('ASCII'))

            flatPixels = [item for sublist in self._imagePixels for item in sublist]

            lines.append(b''.join([i.to_bytes(1, 'big') for i in flatPixels]))
            
            with open(outputPath, 'wb') as outputFile:
                for line in lines:
                    outputFile.write(line)
        else:
            raise ValueError(f'Image object at <{id(self)}> has an invalid magic number ({self._magicNumber}) and its specification is not implemented for saving.')

    def __getMeanIntensityFromPixels(self, pixels: list):
        flatPixels = flatten(pixels)
        return mean(flatPixels)

    def __getMedianIntensityFromPixels(self, pixels: list):
        flatPixels = flatten(pixels)
        return median(flatPixels)

    def __getMaxvalFromPixels(self, pixels: list):
        flatPixels = flatten(pixels)
        return max(flatPixels)

    def __getDimensionsFromPixels(self, pixels: list):
        width = len(pixels[0])
        height = len(pixels)
        return width, height

    def __getHistogramFromPixels(self, pixels: list) -> list:  
        """
            Generates histogram from array of pixels in [[value: int]*width]*height format.
            Only works for PGM files past April/2000. After that, PGM's could have maxval greater than 255.

            Args:
                pixels (list): array of pixels in the image
        """
        histogram = [0] * 256
        flatPixels = flatten(pixels)

        for pixel in flatPixels:
            histogram[pixel] += 1

        return histogram

    def getPixels(self):
        return self._imagePixels
    
    def getMaxval(self):
        return self._maxval

    def getMagicNumber(self):
        return self._magicNumber
    
    def getDimensions(self):
        return self._width, self._height

    def getHistogram(self):
        return self._imageHistogram
    
    def getMeanIntensity(self):
        return int(self._meanIntensity)

    def getMedianIntensity(self):
        return int(self._medianIntensity)
    
    def getSgtThreshold(self):
        return int(self._T)

    def thresholding(self, t=127, save=False, outputPath:str = ''):
        """
            Applies the thresholding filter onto an Image object, considering a initial threshold t.

            Args:
                t (int): initial threshold.
                outputPath (str): Output path for the file.

            Returns:
                A new Image object with the applied threshold.
        """
        newPixels = copy.deepcopy(self._imagePixels)
        
        for row in range(len(newPixels)):
            for item in range(len(newPixels[row])):
                if newPixels[row][item] > t:
                    newPixels[row][item] = 255
                else:
                    newPixels[row][item] = 0

        newImage = Image(
            fromFile = False,
            magicNumber = self._magicNumber,
            imagePixels = newPixels)
        
        if(save):
            newImage.saveToPGM(outputPath)

        return newImage 

    def sgt(self, dt=1, save:bool = False, outputPath:str = ''):
        """
            Applies the thresholding filter using Simple Global Linearization method onto an Image object, considering a dt threshold variation.

            Args:
                dt (int): Maximum variation between previous and new threshold value.
                outputPath (str): Output path for the file.

            Returns:
                A new Image object with the applied threshold.
        """

        newPixels = copy.deepcopy(self._imagePixels)

        T_old = self._meanIntensity
        T_new = T_old

        while True:
            G1 = [] # intensity > T
            G2 = [] # intensity <= T
            for row in range(len(newPixels)):
                for item in range(len(newPixels[row])):
                    if newPixels[row][item] > T_old:
                        G1.append(newPixels[row][item])
                    else:
                        G2.append(newPixels[row][item])

            mean_G1 = mean(G1)
            mean_G2 = mean(G2)

            T_new = int(0.5 * (mean_G1 + mean_G2))

            if( abs(T_new - T_old)) < dt:
                break
            else:
                T_old = T_new

        self._T = T_old
        
        newImage = self.thresholding(t=int(T_old), save=False)
        
        if(save):
            newImage.saveToPGM(outputPath)

        return newImage

    def mean(self, k:int = 3, save:bool = False, outputPath:str = ''):
        """
            Applies the mean filter onto an Image object, considering k x k neighbourhoods.

            Args:
                k (int): Size of the neighbourhood (matrix of k x k), k odd.
                outputPath (str): Output path for the file.

            Returns:
                A new Image object with the applied filter.
        """

        assert k % 2 != 0, "k must be an odd number"
        assert k >= 3, "k must be greater than or equal 3"

        newPixels = copy.deepcopy(self._imagePixels)
        offset = k//2

        for row in range(self._height):
            for col in range(self._width):
                kernel = []

                for x in range(-offset,offset+1):
                    for y in range(-offset,offset+1):
                        i_probe = row+x
                        j_probe = col+y

                        height_inside = (i_probe >= 0 and i_probe < self._height)
                        width_inside = (j_probe >= 0 and j_probe < self._width)
                        is_inside = height_inside and width_inside

                        if is_inside:
                            kernel.append(newPixels[i_probe][j_probe])
                        else:
                            kernel.append(0)
                averageKernel = mean(kernel)
                newPixels[row][col] = int(averageKernel)

        newImage = Image(
            fromFile = False,
            magicNumber = self._magicNumber,
            imagePixels = newPixels 
        )

        if(save):
            newImage.saveToPGM(outputPath)

        return newImage
                
    def median(self, k:int = 3, save:bool = False, outputPath: str = ''):
        """
            Applies the median filter onto an Image object, considering k x k neighbourhoods.

            Args:
                k (int): Size of the neighbourhood (matrix of k x k).
                outputPath (str): Output path for the file.

            Returns:
                A new Image object with the applied filter.
        """

        assert k % 2 != 0, "k must be an odd number"
        assert k >= 3, "k must be greater than or equal 3"

        newPixels = copy.deepcopy(self._imagePixels)
        offset = k//2

        for row in range(self._height):
            for col in range(self._width):
                kernel = []

                for x in range(-offset,offset+1):
                    for y in range(-offset,offset+1):
                        i_probe = row+x
                        j_probe = col+y

                        height_inside = (i_probe >= 0 and i_probe < self._height)
                        width_inside = (j_probe >= 0 and j_probe < self._width)
                        is_inside = height_inside and width_inside

                        if is_inside:
                            kernel.append(newPixels[i_probe][j_probe])
                        else:
                            kernel.append(0)
                medianKernel = median(kernel)
                newPixels[row][col] = int(medianKernel)

        newImage = Image(
            fromFile = False,
            magicNumber = self._magicNumber,
            imagePixels = newPixels 
        )

        if(save):
            newImage.saveToPGM(outputPath)

        return newImage

    def sobel(self, save: bool = False, outputPath: str = ''):
        #partial images
        Gx_img=copy.deepcopy(self._imagePixels)
        Gy_img=copy.deepcopy(self._imagePixels)

        #kernels
        Gx=[[1,0,-1],[2,0,-2],[1,0,-1]]
        Gy=[[1,2,1],[0,0,0],[-1,-2,-1]]

        #resulting img
        filtered_image=copy.deepcopy(self._imagePixels)

        # to generate both imgs, Gx_img and Gy_img
        # the loop is applied two times,
        # Gx_img -> times=0, Gy_times -> times=1
        for times in range(2):
            for line in range(self._height):
                for column in range(self._width):
                    cumulative_sum=0
                    for dx in range(-1,1+1): 
                        for dy in range(-1,1+1):
                            if 0<= line + dx <self._height and 0<= column + dy <self._width:
                                if times==0:
                                    cumulative_sum+=(self._imagePixels[line+dx][column+dy]*Gx[1+dx][1+dy])
                                if times==1:
                                        cumulative_sum+=(self._imagePixels[line+dx][column+dy]*Gy[1+dx][1+dy])
                                        Gy[1+dx][1+dy]
                    if times==0: Gx_img[line][column]=cumulative_sum
                    if times==1: Gy_img[line][column]=cumulative_sum

        bigger_value=0
        for line in range(self._height):
            for column in range(self._width):
                new_px=(Gx_img[line][column]**2+Gy_img[line][column]**2)**0.5
                filtered_image[line][column]=new_px
                #the following line is searching for the bigger pixel inside the img
                if new_px>bigger_value: bigger_value=new_px
        #now the img pixels are normalized to 8bit 
        for line in range(self._height):
            for column in range(self._width):
                filtered_image[line][column]=round(255*filtered_image[line][column]/bigger_value)    

        newImage = Image(
            fromFile = False,
            magicNumber = self._magicNumber,
            imagePixels = filtered_image 
        )

        if(save):
            newImage.saveToPGM(outputPath)

        return newImage

def handleCLI():
    parser = argparse.ArgumentParser(
        prog="POO Image Processor (Project 2)",
        description="Thresholds images and applies filters on them")

    parser.add_argument('--imgpath', 
        help="Path of the image",
        type=str,
        required=True)

    parser.add_argument('--outputpath', 
        help="Path to store the result. If not provided, the program will store the result in the current working directory.",
        type=str,
        required=False)

    parser.add_argument('--op', 
        choices=['thresholding', 'sgt', 'mean', 'median','sobel'],
        help="Operation to be done on the image",
        required=True)

    parser.add_argument('--t', type=int, required=False) # thresholding
    parser.add_argument('--dt', type=float, required=False) # sgt
    parser.add_argument('--k', type=int, required=False) # mean/median
    
    args = parser.parse_args()

    return args

def mainRoutine(args):
    currentDir = os.getcwd()

    if(not os.path.isfile(args.imgpath)):
        raise Exception(f"The provided path '{args.imgpath}' is not a file.")

    parsedPGM = Image(fromFile=True, filePath=args.imgpath)
    
    if(args.op == 'thresholding'):
        output = os.path.join(currentDir, 'thresholding.pgm') if not args.outputpath else os.path.join(args.outputpath, 'thresholding.pgm')
        newPGM = parsedPGM.thresholding(t=args.t, save=True, outputPath=output)

    elif(args.op == 'sgt'):
        output = os.path.join(currentDir, 'sgt.pgm') if not args.outputpath else os.path.join(args.outputpath, 'sgt.pgm')
        newPGM = parsedPGM.sgt(dt=args.dt, save=True, outputPath=output)

        # As specified by the professor, we'll only print output to console in the sgt case.

        print(f"magic_number {parsedPGM.getMagicNumber()}")
        print(f"dimensions {parsedPGM.getDimensions()}")    
        print(f"maxval {parsedPGM.getMaxval()}")
        print(f"mean {parsedPGM.getMeanIntensity()}")
        print(f"median {parsedPGM.getMedianIntensity()}")
        print(f"T {parsedPGM.getSgtThreshold()}")

    elif(args.op == 'mean'):
        output = os.path.join(currentDir, 'mean.pgm') if not args.outputpath else os.path.join(args.outputpath, 'mean.pgm')
        newPGM = parsedPGM.mean(k=args.k, save=True, outputPath=output)

    elif(args.op == 'median'):
        output = os.path.join(currentDir, 'median.pgm') if not args.outputpath else os.path.join(args.outputpath, 'median.pgm')
        newPGM = parsedPGM.median(k=args.k, save=True, outputPath=output)

    elif (args.op == 'sobel'):
        output = os.path.join(currentDir, 'sobel.pgm') if not args.outputpath else os.path.join(args.outputpath, 'sobel.pgm')
        newPGM = parsedPGM.sobel(save=True, outputPath=output)

    return

def meanParamVerify(params):
    for dic in params:
        if(dic['param'] == '--k'):
            try: 
                int(dic['value'])
            except:
                raise Exception("--k parameter must be an integer")

def medianParamVerify(params):
    for dic in params:
        if(dic['param'] == '--k'):
            try: 
                int(dic['value'])
            except:
                raise Exception("--k parameter must be an integer")

def thresholdingParamVerify(params):
    for dic in params:
        if(dic['param'] == '--t'):
            try: 
                int(dic['value'])
            except:
                raise Exception("--t parameter must be an integer")

def sgtParamVerify(params):
    for dic in params:
        if(dic['param'] == '--dt'):
            try: 
                float(dic['value'])
            except:
                raise Exception("--dt parameter must be a float")

def sobelParamVerify(params):
    if(len(params) > 0):
        raise Exception("sobel operation can't take arguments.")

def customParser():
    args = sys.argv[1:]

    help_cmds = ['-h', '--h', '--help', '-help', '--help']
    help_msg="""POO Image Processor (Project 3)
To show this help message, use -h/--h/-help/--help.

* Required arguments:
    --imgpath: Path of the source image
    --op: operation or chain of operations and its parameters to be applied to the Image.

* Optional arguments:
    --outputpath: Directory to store the result. If not provided, the program will store the result in the current working directory.

* Chain of operations:
    In the --op argument, you can pass a chain of operations to be done on the image, followed by its parameters.
    The operations will be executed in sequence, using the passed parameters.
    If no parameter is specificied in the specific operation, the default parameters will be used for that specific operation.

    It is required to provide at lest one valid operation on --op. All operations arguments are optional.

    The program won't throw an error if you pass a unallowed operation, but it will be desconsidered.

    List of allowed operations and its allowed parameters:

    mean
        parameters:
            --k: 

    median
        parameters:
            --k: 

    thresholding
        parameters:
            --t: 

    sgt
        parameters:
            --dt: 

    sobel
        parameters: none
"""

    if(len(args) == 0):
        print(help_msg)
        sys.exit()

    for i in help_cmds:
        if i in args:
            print(help_msg)
            sys.exit()

    if ('--imgpath' not in args):
        raise Exception(f"The argument {'--imgpath'} must be passed. See -h for help.")
    if('--op' not in args):
        raise Exception(f"The argument {'--op'} must be passed. See -h for help.")
    
    imgpath_index = None
    op_index = None
    outputpath_index = None

    outputpath='./'

    for i in range(len(args)):
        if(args[i] == '--imgpath'):
            imgpath_index = i
        if(args[i] == '--op'):
            op_index = i
        if(args[i] == '--outputpath'):
            outputpath_index = i

    imgpath = args[imgpath_index+1]

    if(not os.path.isfile(imgpath)):
        raise Exception(f"The provided imgpath '{imgpath}' does not exist or is not a file.")

    if(outputpath_index):
        outputpath = args[outputpath_index+1]

        if(not os.path.isdir(outputpath)):
            raise Exception(f"The provided outputpath '{outputpath}' does not exist or is not a directory.")
    
    del_indices = [imgpath_index, imgpath_index+1, op_index]

    if(outputpath_index):
        del_indices.append(outputpath_index)
        del_indices.append(outputpath_index+1)

    op_chain = [i for j, i in enumerate(args) if j not in del_indices]

    chainSpecs = {
        'mean': {
            'params': ['--k'],
            'verify': meanParamVerify,
        },
        'median': {
            'params': ['--k'],
            'verify': medianParamVerify,
        },
        'thresholding':{
            'params': ['--t'],
            'verify': thresholdingParamVerify,
        },
        'sgt': {
            'params': ['--dt'],
            'verify': sgtParamVerify,
        },
        'sobel': {
            'params': None,
            'verify': sgtParamVerify,
        },
    }

    processed_Chain = []
    print(op_chain)
    for i in range(len(op_chain)):
        if(op_chain[i] in chainSpecs.keys()):
            currentOp = chainSpecs[op_chain[i]]
            currentParams = []

            if(op_chain[i] != 'sobel'):
                if(len(op_chain) > i+1):
                    if(op_chain[i+1] in currentOp['params']):
                        currentParams.append({'param': op_chain[i+1], 'value': op_chain[i+2]})
                        currentOp['verify'](currentParams) # Throws error if parameters are invalid
            
            processed_Chain.append({'op': op_chain[i], 'params': currentParams})

    if(len(processed_Chain) == 0):
        raise Exception("No valid operations were provided. See -h for help.")

    return processed_Chain, imgpath, outputpath

def mainRoutine2():
    chain, imgpath, outputdir = customParser()
    print(chain)

    outputpath=''
    

    if(outputdir):
        outputpath = os.path.join(outputdir, 'output.pgm')
    else:
        outputpath = os.path.join(os.getcwd(), 'output.pgm')

    startPGM = Image(fromFile=True, filePath=imgpath)

    for spec in chain:
        if(spec['op'] == 'mean'):
            print('running mean')
            parameters = {
                'k': None
            }
            if(spec['params'] != []):
                for dic in spec['params']:
                    if(dic['param'] == '--k'):
                        parameters['k'] = dic['value'] 
            
            if(parameters['k']):
                print(f"mean has k= {parameters['k']}")
                startPGM = startPGM.mean(k=int(parameters['k']), save=False)
            else:
                 startPGM = startPGM.mean(save=False)

        elif(spec['op'] == 'median'):
            parameters = {
                'k': None
            }
            if(spec['params'] != []):
                for dic in spec['params']:
                    if(dic['param'] == '--k'):
                        parameters['k'] = dic['value'] 
            
            if(parameters['k']):
                startPGM = startPGM.median(k=int(parameters['k']), save=False)
            else:
                 startPGM = startPGM.median(save=False)

        elif(spec['op'] == 'thresholding'):
            parameters = {
                't': None
            }
            if(spec['params'] != []):
                for dic in spec['params']:
                    if(dic['param'] == '--t'):
                        parameters['t'] = dic['value'] 
            
            if(parameters['t']):
                startPGM = startPGM.thresholding(t=int(parameters['t']), save=False)
            else:
                 startPGM = startPGM.thresholding(save=False)

        
        elif(spec['op'] == 'sgt'):
            parameters = {
                'dt': None
            }
            if(spec['params'] != []):
                for dic in spec['params']:
                    if(dic['param'] == '--dt'):
                        parameters['dt'] = dic['value'] 
            
            if(parameters['dt']):
                startPGM = startPGM.sgt(dt=float(parameters['dt']), save=False)
            else:
                startPGM = startPGM.sgt(save=False)
        
        elif(spec['op'] == 'sobel'):
            print('running sobel')
            startPGM = startPGM.sobel(save=False)

    head, tail = os.path.split(imgpath)
    print(f"image_name : {tail}")
    for specs in chain:
        print(f"op : {specs['op']}")
        if specs['params'] != []:
            for dic in specs['params']:
                print(f"\t{ dic['param'].replace('--', '')} : {dic['value']}")
    
    startPGM.saveToPGM(outputpath)

if __name__ == '__main__':
    # This does a graceful shutdown without showing tracebacks, only error messages.
    #args = handleCLI()
    try:
        mainRoutine2()
    except BaseException as error:
        print(f"Error: {str(error)}")
    finally:
        sys.exit()