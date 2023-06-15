class Image:
    def __init__(self):
        pass

    def thresholding(self, t=127):
        pass
    def sgt(self, dt=1):
        pass
    def mean(self, k=3):
        """
            Applies the mean filter onto an Image object, considering 3x3 neighbourhoods.

            Args:
                param1: This is the first param.
                param2: This is a second param.

            Returns:
                This is a description of what is returned.

            Raises:
                KeyError: Raises an exception.
        """

        # save image in path
        return Image()

    def median(self, k=3):
        """
            Applies the median filter onto an Image object, considering 3x3 neighbourhoods.

            Args:
                param1: This is the first param.
                param2: This is a second param.

            Returns:
                This is a description of what is returned.

            Raises:
                KeyError: Raises an exception.
        """

        # save image in path
        return Image()


def main():
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog="POO Image Processor (Projeto 2)",
        description="Thresholds images and applies filters on them")
    parser.add_argument('--imgpath', help='Path of the image')
    parser.add_argument('--op',)
    parser.add_argument('--t',)
    parser.add_argument('--dt',)
    parser.add_argument('--k',)
    main()