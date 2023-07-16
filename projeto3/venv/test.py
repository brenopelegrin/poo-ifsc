import argparse
import sys
import os

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
    for i in range(len(op_chain)):
        if(op_chain[i] in chainSpecs.keys()):
            currentOp = chainSpecs[op_chain[i]]
            currentParams = []

            if( len(op_chain) > i+1 and op_chain[i+1] in currentOp['params']):
                currentParams.append({'param': op_chain[i+1], 'value': op_chain[i+2]})
                currentOp['verify'](currentParams) # Throws error if parameters are invalid
            
            processed_Chain.append({'op': op_chain[i], 'params': currentParams})

    if(len(processed_Chain) == 0):
        raise Exception("No valid operations were provided. See -h for help.")

    return processed_Chain, imgpath, outputpath


def mainRoutine():
    chain, imgpath, outputdir = customParser()

    outputpath=''

    if(outputdir):
        outputpath = os.path.join(outputdir, 'output.pgm')
    else:
        outputpath = os.path.join(os.getcwd(), 'output.pgm')

    startPGM = Image(fromFile=True, filePath=imgpath)

    for spec in chain:
        if(spec['op'] == 'mean'):
            parameters = {
                'k': None
            }
            if(spec['params'] != []):
                for dic in spec['params']:
                    if(dic['param'] == '--k'):
                        parameters['k'] = dic['value'] 
            
            if(parameters['k']):
                startPGM = startPGM.mean(k=parameters['k'], save=False)
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
                startPGM = startPGM.median(k=parameters['k'], save=False)
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
                startPGM = startPGM.thresholding(t=parameters['t'], save=False)
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
                startPGM = startPGM.sgt(dt=parameters['dt'], save=False)
            else:
                startPGM = startPGM.sgt(save=False)
        
        elif(spec['op'] == 'sobel'):
            startPGM = startPGM.sobel(save=False)

    head, tail = os.path.split(imgpath)
    print(f"image_name : {tail}")
    for i in specs:
        print(f"op : {specs['op']}")
        if specs['params'] != []:
            for dic in specs['params']:
                print(f"\t{ dic['param'].replace('--', '')} : {dic['value']}")
    
    startPGM.saveToPGM(outputpath)

if __name__ == '__main__':
    mainRoutine()


