import argparse
import sys
import os
# Disclaimer about the packages usage:
# The "os" package will be used to join paths and filenames and provide compatibility between UNIX-based systems and DOS-based systems.
# The "sys" package will be used to provide clean exit for production environment.
# The "argparse" package will be used to parse CLI arguments.

class Image:
    def __init__(self, fromFile: bool = False, filePath: str = '', magicNumber: str = '', imagePixels: list = []):

        if(fromFile):
            assert filePath != '', "To create an Image from file, pass the filePath argument."

            self._fromFile = True
            self._filePath = filePath

            fileData = self.__readFromPGM(self._filePath)

            self._magicNumber = fileData['type']
            self._imagePixels = fileData['pixels']
        else:
            assert magicNumber != '', "To manually create an Image, pass the magicNumber argument."
            assert imagePixels != [], "To manually create an Image, pass the imagePixels argument."

            self._fromFile = False
            self._filePath = ''

            self._magicNumber = magicNumber
            self._imagePixels = imagePixels

        self._width, self._height = self.__getDimensionsFromPixels(self._imagePixels)
        self._maxval = self.__getMaxvalFromPixels(self._imagePixels)
        self._imageHistogram = self.__getHistogramFromPixels(self._imagePixels)

    def __readFromPGM(self, filePath):
        """
            Reads a PGM file and convert its content into usable data.
            Only works for PGM files past July/2000. After that, PGM's could have more than one image in one file.

            Args:
                filePath (str): path to the pgm file
        """
        
        lines = []
        with open(filePath, 'r', encoding='UTF-8') as file:
            while line := file.readline():
                currentLine = line.rstrip()
                if('#' not in currentLine and currentLine != ''):
                    lines.append(line.rstrip())
        data = {
            'type': lines[0],
            'width': int(lines[1].split()[0]),
            'height': int(lines[1].split()[1]),
            'maxval': int(lines[2]),
            'pixels': []
        }

        dataLines = lines[3:]
        
        for line in dataLines:
            row = [int(x) for x in line.split()]
            data['pixels'].append(row)

        return data
    
    def saveToPGM(self, outputPath):
        if(self._magicNumber == 'P2'):
            lines = []
            lines.append(f'{_magicNumber} \n')
            lines.append(f'{self._width} {self._height} \n')
            lines.append(f'{self._maxval} \n')
            for row in self._imagePixels:
                stringRow = ' '.join([str(item) for item in row])+' \n'
                lines.append(stringRow)
            
            with open(outputPath, 'w') as outputFile:
                for line in lines:
                    outputFile.write(line)

        elif(self._magicNumber == 'P5'):
            lines = []
            lines.append(f'{_magicNumber} \n')
            lines.append(f'{self._width} {self._height} \n')
            lines.append(f'{self._maxval} \n')
            for row in self._imagePixels:
                stringRow = ' '.join([str(item.to_bytes()) for item in row])+' \n'
                lines.append(stringRow)
            
            with open(outputPath, 'w') as outputFile:
                for line in lines:
                    outputFile.write(line)
        else:
            raise ValueError(f'Image <{self.__name__}> has an invalid magic number ({self._magicNumber}) its specification is not implemented for saving.')

    def __getMaxvalFromPixels(self, pixels: list):
        flatPixels = [item for sublist in pixels for item in sublist]
        return max(flatPixels)

    def __getDimensionsFromPixels(self, pixels: list):
        width = len(pixels[0])
        height = len(pixels)
        return width, height

    def __getHistogramFromPixels(self, pixels: list):  
        """
            Generates histogram from array of pixels in [[value: int]*width]*height format.
            Only works for PGM files past April/2000. After that, PGM's could have maxval greater than 255.

            Args:
                pixels (list): array of pixels in the image
        """
        histogram = [0] * 256
        flatPixels = [item for sublist in pixels for item in sublist]

        for pixel in flatPixels:
            histogram[pixel] += 1

        return histogram

    def thresholding(self, t=127, outputPath):
        newPixels = self._imagePixels.copy()
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
        
        newImage.saveToPGM(outputPath)

        return newImage 

    def sgt(self, dt=1, outputPath):
        T = [127]
        i=-1
        newPixels = self._imagePixels.copy()

        while T[i+1] - T[i] >= dt:
            G1 = [] # intensity > T
            G2 = [] # intensity <= T
            for row in range(len(newPixels)):
                for item in range(len(newPixels[row])):
                    if newPixels[row][item] > T[i]:
                        G1.append(newPixels[row][item])
                    else:
                        G2.append(newPixels[row][item])
            mean_G1 = sum(G1)/len(G1)
            mean_G2 = sum(G2)/len(G2)
            T.append(0.5 * mean_G1 + mean_G2)
            i+=1

        self._T = T[-1]

        for row in range(len(newPixels)):
            for item in range(len(newPixels[row])):
                if newPixels[row][item] > self._T:
                    newPixels[row][item] = 255
                else:
                    newPixels[row][item] = 0

        newImage = Image(
            fromFile = False,
            magicNumber = self._magicNumber,
            imagePixels = newPixels)
        
        newImage.saveToPGM(outputPath)

        return newImage

    def mean(self, k:int = 3, outputPath: str):
        """
            Applies the mean filter onto an Image object, considering k x k neighbourhoods.

            Args:
                k (int): Size of the neighbourhood (matrix of k x k).
                outputPath (str): Output path for the file.

            Returns:
                A new Image object with the applied filter.
        """
        assert k % 3 == 0, "k must be an odd number"
        assert k >= 3, "k must be greater than or equal 3"

        newPixels = self._imagePixels.copy()
        meanPixels = newPixels.copy()
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
                    for j_prime in range(j-maxOffset, j+maxOffset+1)
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
                            neighbourhoodMatrix[maxOffest][maxOffset] = centralPixel

                
                flatNeighbourhoodMatrix = [item for sublist in neighbourhoodMatrix for item in sublist]
                meanFromNeighbourhood = round(sum(flatNeighbourhoodMatrix)/len(flatNeighbourhoodMatrix))
                meanPixels[i][j] = meanFromNeighbourhood
        
        newImage = Image(
            fromFile = False,
            magicNumber = self._magicNumber,
            imagePixels = meanPixels
        )

        newImage.saveToPGM(outputPath)

        return newImage
        

    def median(self, k:int = 3, outputPath: str):
        """
            Applies the median filter onto an Image object, considering k x k neighbourhoods.

            Args:
                k (int): Size of the neighbourhood (matrix of k x k).
                outputPath (str): Output path for the file.

            Returns:
                A new Image object with the applied filter.
        """

        # save image in path
        pass

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

def mainRoutine():
    args = handleCLI()
    inputPath = args.imgpath
    outputPath = args.outputpath
    operation = args.op
    t = args.t
    dt = args.dt
    k = args.k

    currentDir = os.getcwd()
    parsedPGM = Image(fromFile=True, filePath=inputPath)
    
    if(operation == 'thresholding'):
        output = os.path.join(currentDir, 'thresholding.pgm') if not outputPath else os.path.join(outputPath, 'thresholding.pgm')
        newPGM = parsedPGM.thresholding(t, output)

    elif(operation == 'sgt'):
        output = os.path.join(currentDir, 'sgt.pgm') if not outputPath else os.path.join(outputPath, 'sgt.pgm')
        newPGM = parsedPGM.sgt(t, output)

    elif(operation == 'mean'):
        output = os.path.join(currentDir, 'mean.pgm') if not outputPath else os.path.join(outputPath, 'mean.pgm')
        pass

    elif(operation == 'median'):
        output = os.path.join(currentDir, 'median.pgm') if not outputPath else os.path.join(outputPath, 'median.pgm')
        pass

    return

if __name__ == '__main__':
    mainRoutine()
    sys.exit()