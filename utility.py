class Check:

    def positive_arguments(*Args):
        """ Test if all the arguments are positive; if not the returns -1.
        If almost one argument isn't a numeric then returns 1.
        Parameters: (**Args) a list of numeric arguments"""
        for e in Args:
            if not e.isnumeric():
                return 1
            if e<0:
                return -1
        return 0