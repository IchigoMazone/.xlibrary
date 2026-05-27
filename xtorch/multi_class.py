
class Size(tuple):

    def __new__(cls, iterable = ()):
        return super(Size, cls).__new__(cls, iterable)
    
    def __repr__(self):
        return f"xtorch.Size({list(self)})"