#################################################
# Projeto 1 - Programação Orientada a Objetos   #
# Prof. Gonzalo Travieso                        #
# IFSC/USP                                      #
# Aluno: Breno Henrique Pelegrin da Silva       #
# N. USP: 13687303                              #
# Data: 20/04/2023                              #
#################################################

import sys

def handleArguments():
    if(len(sys.argv) != 3):
        print("Erro: você precisa passar dois argumentos: o nome do arquvio e o número de bins. Exemplo: python main.py pgm_filename number_of_bins")
        sys.exit()

    pgmFilePath = sys.argv[1]
    bins = sys.argv[2]

    if(not bins.replace('-', '', 1).isdigit()):
        print("Erro: o número de bins deve ser um inteiro.")
        sys.exit()
    else:
        bins = int(bins)
        if(bins) <= 0:
            print(f"Erro: não é possível gerar {bins} bins.")
            sys.exit()

    # The project description says explicitly that we must only use the "sys" package.
    # Hence, we'll assume that the file exists, since we can't use the "os" package.
    return(pgmFilePath, bins)

def extractFileData(pgmFilePath):
    lines = []
    with open(pgmFilePath, 'r', encoding='UTF-8') as file:
        while line := file.readline():
            currentLine = line.rstrip()
            # Verify if line isn't a comment AND line isn't empty
            if('#' not in currentLine) and currentLine != '':
                lines.append(line.rstrip())
    data = {
        'type': lines[0],
        'width': int(lines[1].split()[0]),
        'height': int(lines[1].split()[1]),
        'maxval': int(lines[2]),
        'schema': None,
        'pixels': []
    }

    dataLines = lines[3:]
    if(len(dataLines) == 1):
        # The pixels are in the "1-line" format.
        data['schema'] = '1-line'
    elif(len(dataLines) == data['height']):
        # The pixels are in the "height x width" format.
        data['schema'] = 'height-width'
    else:
        # The pixels are in other format.
        data['schema'] = 'other'
    
    for line in dataLines:
        # Get all the pixels data from file
        [data['pixels'].append(float(x)) for x in line.split()]

    return(data)

def getProbabilityDistributions(pixels, intervals):
    densityByInterval = {}
    for interval in intervals:
        start = float(interval[0])
        end = float(interval[1])
        pixelsInInterval = [x for x in pixels if x >= start and x < end]
        densityByInterval[f"[{'%.2f' % start}, {'%.2f' % end})"] = {
            'pixels': len(pixelsInInterval),
            'density': len(pixelsInInterval)/len(pixels) # density = number_of_pixels_in_interval/number_of_total_pixels
        }
        
    return(densityByInterval)

if __name__ == '__main__':
    pgmFilePath, bins = handleArguments()
    data = extractFileData(pgmFilePath) 

    # Computers starts counting from 0, but humans start counting from 1, so we need to use data['maxval']+1
    binSize = (data['maxval']+1)/bins

    if(bins > data['maxval']+1):
        print(f"Erro: número de bins pedido {bins}, mas {data['maxval']} é o valor máximo de intensidade na imagem.")
        sys.exit()

    intervals = []
    for i in range(bins):
        intervals.append((i*binSize, (i+1)*binSize))
    distributions = getProbabilityDistributions(data['pixels'], intervals)
    for interval in distributions:
        print(f"{interval} {'%d' % distributions[interval]['pixels']} {'%.5f' % distributions[interval]['density']}")

