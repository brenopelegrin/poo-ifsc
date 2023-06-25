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
    lst = sorted(array)  # Sort the list first
    if len(lst) % 2 == 0:  # Checking if the length is even
        # Applying formula which is sum of middle two divided by 2
        return (lst[len(lst) // 2] + lst[(len(lst) - 1) // 2]) / 2
    else:
        # If length is odd then get middle value
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

        self._meanIntensity = self.__getMeanIntensityFromPixels(self._imagePixels)
        self._medianIntensity = self.__getMedianIntensityFromPixels(self._imagePixels)
        self._width, self._height = self.__getDimensionsFromPixels(self._imagePixels)
        self._imageHistogram = self.__getHistogramFromPixels(self._imagePixels)

        self._T = None

    def __readFromPGM(self, filePath: str) -> dict:  
        """
            Reads a PGM file and convert its content into usable data.
            Only works for PGM files past July/2000. After that, PGM's could have more than one image in one file.

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
            raise Exception(f'File <{inputPath}> has invalid magic number (neither P5 or P2).')
    
    def saveToPGM(self, outputPath):
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
            raise ValueError(f'Image <{self.__name__}> has an invalid magic number ({self._magicNumber}) and its specification is not implemented for saving.')

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

    def thresholding(self, t=127, save=False, outputPath:str = ''):
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
        newPixels = copy.deepcopy(self._imagePixels)

        T_old = self.getMeanIntensity()
        T_new = T_old+dt
        while True:
            G1 = [] # intensity > T
            G2 = [] # intensity <= T
            for row in range(len(newPixels)):
                for item in range(len(newPixels[row])):
                    if newPixels[row][item] > T_old:
                        G1.append(newPixels[row][item])
                    else:
                        G2.append(newPixels[row][item])
            mean_G1 = sum(G1)/len(G1)
            mean_G2 = sum(G2)/len(G2)
            T_new = (0.5 * (mean_G1 + mean_G2))
            if( abs(T_new - T_old)) < dt:
                break
            else:
                T_old = T_new

        self._T = T_new

        newImage = self.thresholding(t=self._T, save=False)
        
        if(save):
            newImage.saveToPGM(outputPath)

        return newImage

    def mean(self, k:int = 3, save:bool = False, outputPath: str = ''):
        """
            Applies the mean filter onto an Image object, considering k x k neighbourhoods.

            Args:
                k (int): Size of the neighbourhood (matrix of k x k).
                outputPath (str): Output path for the file.

            Returns:
                A new Image object with the applied filter.
        """
        assert k % 2 != 0, "k must be an odd number"
        assert k >= 3, "k must be greater than or equal 3"

        newPixels = copy.deepcopy(self._imagePixels)
        meanPixels = copy.deepcopy(newPixels)
        rows = len(newPixels) - 1
        cols = len(newPixels[0]) - 1
        # Considering a square matrix k x k centered in the interest element, 
        # In the diagonals, we will have (k-1)/2 elements before and after the interest element.
        maxOffset = int(0.5*(k-1))

        # Iterate through each element of newPixels and calculate its new value based on neighbourhood matrix.
        for i in range(rows+1):
            for j in range(cols+1):
                neighbourhoodMatrix = [[0]*k]*k
                interestElement = newPixels[i][j]
                
                # i_prime and j_prime are indexes relative to interestElement, varying from -maxOffset to +maxOffset (the +1 is because of range implementation)
                for i_prime in range(i-maxOffset, i+maxOffset+1):
                    for j_prime in range(j-maxOffset, j+maxOffset+1):
                        # nbhd_i and nbhd_j retrives i and j for neighbourhoodMatrix
                        nbhd_i = i_prime - i
                        nbhd_j = j_prime - j
                        if i_prime != i and j_prime != j:
                            try:
                                neighbourPixel = newPixels[i_prime][j_prime]
                                neighbourhoodMatrix[nbhd_i][nbhd_j] = neighbourPixel
                            except:
                                neighbourPixel = 0
                                neighbourhoodMatrix[nbhd_i][nbhd_j] = neighbourPixel
                        else:
                            centralPixel = 0
                            neighbourhoodMatrix[maxOffset][maxOffset] = centralPixel

                
                flatNeighbourhoodMatrix = flatten(neighbourhoodMatrix)
                meanFromNeighbourhood = round(mean(flatNeighbourhoodMatrix))
                meanPixels[i][j] = meanFromNeighbourhood
        
        newImage = Image(
            fromFile = False,
            magicNumber = self._magicNumber,
            imagePixels = meanPixels
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
        medianPixels = copy.deepcopy(newPixels)
        rows = len(newPixels) - 1
        cols = len(newPixels[0]) - 1
        # Considering a square matrix k x k centered in the interest element, 
        # In the diagonals, we will have (k-1)/2 elements before and after the interest element.
        maxOffset = int((k-1)/2)

        # Iterate through each element of newPixels and calculate its new value based on neighbourhood matrix.
        for i in range(rows+1):
            for j in range(cols+1):
                neighbourhoodMatrix = [[0]*k]*k
                interestElement = newPixels[i][j]
                
                # i_prime and j_prime are indexes relative to interestElement, varying from -maxOffset to +maxOffset (the +1 is because of range implementation)
                for i_prime in range(i-maxOffset, i+maxOffset+1):
                    for j_prime in range(j-maxOffset, j+maxOffset+1):
                        # nbhd_i and nbhd_j retrives i and j for neighbourhoodMatrix
                        nbhd_i = i_prime - i
                        nbhd_j = j_prime - j
                        if i_prime != i and j_prime != j:
                            try:
                                neighbourPixel = newPixels[i_prime][j_prime]
                                neighbourhoodMatrix[nbhd_i][nbhd_j] = neighbourPixel
                            except:
                                neighbourPixel = 0
                                neighbourhoodMatrix[nbhd_i][nbhd_j] = neighbourPixel
                        else:
                            centralPixel = 0
                            neighbourhoodMatrix[maxOffset][maxOffset] = centralPixel

                
                flatNeighbourhoodMatrix = flatten(neighbourhoodMatrix)
                
                medianFromNeighbourhood = median(flatNeighbourhoodMatrix)

                medianPixels[i][j] = medianFromNeighbourhood
        
        newImage = Image(
            fromFile = False,
            magicNumber = self._magicNumber,
            imagePixels = medianPixels
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
        choices=['thresholding', 'sgt', 'mean', 'median'],
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
        raise Exception(f"The provided path <{args.imgpath}> is not a file.")

    parsedPGM = Image(fromFile=True, filePath=args.imgpath)
    if(args.dt):
        # This triggers the parsedPGM sgt method to save self._T attribute on parsedPGM object
        parsedPGM.sgt(dt=args.dt, save=False)   
    else:
        parsedPGM.sgt(save=False)
    
    if(args.op == 'thresholding'):
        output = os.path.join(currentDir, 'thresholding.pgm') if not args.outputpath else os.path.join(args.outputpath, 'thresholding.pgm')
        newPGM = parsedPGM.thresholding(t=args.t, save=True, outputPath=output)

    elif(args.op == 'sgt'):
        output = os.path.join(currentDir, 'sgt.pgm') if not args.outputpath else os.path.join(args.outputpath, 'sgt.pgm')
        newPGM = parsedPGM.sgt(dt=args.dt, save=True, outputPath=output)

    elif(args.op == 'mean'):
        output = os.path.join(currentDir, 'mean.pgm') if not args.outputpath else os.path.join(args.outputpath, 'mean.pgm')
        newPGM = parsedPGM.mean(k=args.k, save=True, outputPath=output)

    elif(args.op == 'median'):
        output = os.path.join(currentDir, 'median.pgm') if not args.outputpath else os.path.join(args.outputpath, 'median.pgm')
        newPGM = parsedPGM.median(k=args.k, save=True, outputPath=output)

    print(f"magic_number {newPGM.getMagicNumber()}")
    print(f"dimensions {newPGM.getDimensions()}")    
    print(f"maxval {newPGM.getMaxval()}")
    print(f"mean {parsedPGM.getMeanIntensity()}")
    print(f"median {parsedPGM.getMedianIntensity()}")
    print(f"T {parsedPGM.getSgtThreshold()}")

    return

if __name__ == '__main__':
    # This does a graceful shutdown without showing tracebacks, only error messages.
    args = handleCLI()
    try:
        mainRoutine(args)
    except BaseException as error:
        print(f"Error: {str(error)}")
    finally:
        sys.exit()