import operator, itertools, copy
from ._logiccore import LogicCore

class DVec(LogicCore):
    """
    DVec is a dense vector, all functionality uses native python.
    To initialize a vector:

    DVec(data, [dtype=None], [orientation='c'])
    
    Parameters
    ----------
    data : List((int, float, complex))
    dtype : {int, float, complex}, optional
        If this is given all items are cast to this type,
        else the first item of list is used as dtype.
    orientation: {'r', 'c'}, optional
        'r' is a row vector, 'c' is a column vector.

    Returns
    -------
    X : Dvec

    DVec is at its core a simple python list, but changes some behaviour to act more like a vector.
    This means that (DVec) + (DVec) is element wise addition instead of the normal extend behaviour.
    Besides addition it supports element wise:
        subtraction (-)
        muliplication (*)
        division (/)
        power (**)
        all (in)equalities (<, <=, ==, >=, >, !=)
        absolute (abs(x))
    Furthermore it supports the dot product with X @ Y.

    The vector supports some fancy printing by print(DVec)
    """
    def __init__(self, data, dtype=None, orientation='c') -> None:
        self._format = "dvec"
        self.orientation = orientation
        if not data:
            return

        if not isinstance(data, list): raise TypeError(f"Vector data must be an non empty list")
        if dtype is None:
            dtype = type(data[0])
        self.dtype = dtype

        if not dtype in (int, float, complex, bool): raise TypeError(f"Data must be of type int, float or complex, not {dtype}")

        try:
            self.data = list(map(self.dtype, data))
        except (ValueError, TypeError)  as e:
            raise TypeError(f"Not all data can be converted into {self.dtype.__name__}, {e}")

        self.length = len(data)

    
    def __len__(self):
        return self.length

    def __add__(self, other):
        return self.__match_operator(other, operator.__add__)
    
    __iadd__ = __add__
    __radd__ = __add__
    
    def __sub__(self, other):
        return self.__match_operator(other, operator.__sub__)
    
    __isub__ = __sub__
    def __rsub__(self, other):
        return (-1 *self).__match_operator(other, operator.__sub__)
    
    def __mul__(self, other):
        return self.__match_operator(other, operator.__mul__)
        
    __rmul__ = __mul__

    def __matmul__(self, other):
        if not isinstance(other, DVec):
            return NotImplemented
        return sum(self * other)

    def __truediv__(self, other):
        return self.__match_operator(other, operator.__truediv__)
        
    def __rtruediv__(self, other) :
        try:
            self = self ** -1
        except ZeroDivisionError:
            raise ZeroDivisionError("Can not divide by 0")
        return self.__match_operator(other, operator.__mul__)

    def __mod__(self, other):
        return self.__match_operator(other, operator.__mod__)

    def __pow__(self, other):
        return self.__match_operator(other, operator.__pow__)
    
    def __lt__(self, other):
        return self.__match_operator(other, operator.__lt__)
    
    def __le__(self, other):
        return self.__match_operator(other, operator.__le__)
    
    def __eq__(self, other):
        return self.__match_operator(other, operator.__eq__)
    
    def __ne__(self, other):
        return self.__match_operator(other, operator.__ne__)
    
    def __ge__(self, other):
        return self.__match_operator(other, operator.__ge__)
    
    def __gt__(self, other):
        return self.__match_operator(other, operator.__gt__)
    
    __rlt__ = __ge__
    __rle__ = __gt__
    __req__ = __eq__
    __rne__ = __ne__
    __rge__ = __lt__
    __rgt__ = __le__

    def __match_operator(self, other, opp):
        if isinstance(other, DVec):
            if self.length != other.length:
                raise ValueError(f"Vector lengths {self.length} and {other.length} do not match")
            opp_data = [opp(s, o) for s,o in zip(self.data, other.data)]

        elif isinstance(other, (float, int, complex)):
            opp_data = [opp(s, other) for s in self.data]

        elif hasattr(other, "_format"):
            return NotImplemented
        else:
            return NotImplementedError
        return DVec(data=opp_data, orientation=self.orientation)

    def __abs__(self):
        return self.__self_operator(operator.__abs__)
 
    def __self_operator(self, opp):
        opp_data = list(map(opp, self.data))
        return DVec(opp_data, orientation=self.orientation)

    def round(self, r=2):
        """
        DVec.round(r):
        Rounds all elements in data to r digits.
        """
        r_data = [round(i,r) for i in self.data]
        return DVec(data=r_data)
    
    def __getitem__(self, key):
        get_data = self._get_item_logic(key)
        if isinstance(get_data, list):
            return DVec(get_data, orientation=self.orientation)
        return get_data
    
    def __setitem__(self, key, val):
        return self.data.__setitem__(key, val)
    
    def index(self, item):
        """
        DVec.index(item):
        Gets the index of the item if it is in the list,
        else returns ValueError.
        """
        return self.data.index(item)

    @property
    def shape(self):
        """
        DVec.shape
        The 2D shape of the vector,
        mostly used for compatibility with DMatrix

        returns
        -------
        (1, M) or (M, 1)
        Depending on orientation
        
        """
        if self.orientation == 'c':
            return (self.length, 1)
        else:
            return (1, self.length)       

    @property
    def T(self):
        """
        DVec.T
        Transposes the vector.

        returns
        -------
        A copy with transposed orientation.
        """
        new = copy.deepcopy(self)
        if new.orientation == 'c':
            new.orientation = 'r'
        else:
            new.orientation = 'c'
        return new

    def _get_str_items(self):
        if self.length > 6:
            str_items =  list(map(str, self.data[:3] + self.data[-3:]))
        else:
            str_items = list(map(str, self.data))
        return str_items, max(map(len, str_items))
    
    @classmethod
    def arange(cls, *args):
        """
        DVec.arange([start], stop, [step])
        
        Parameters
        ----------
        start: int,
            starting point.
        stop: int,
            stopping point.
        step: int,
            step size.

        Returns
        -------
        X : Dvec
            A column vector with items in the given range.
        """
        return DVec(list(range(*args)))
       
    
    @classmethod
    def _format_row_str(cls, p_data, row_length, items=True):
        if row_length >6:
            return "[" + ", ".join(p_data[:3]) + f" --{(row_length - 6 if items else '-')}-- " + ", ".join(p_data[-3:]) + "]"
        else:
            return "[" + ", ".join(p_data) + "]"
    
    @classmethod
    def _format_col_str(cls, p_data, col_length, items=True):
        if col_length>6:
            return "\n".join(["|"+i+"|" for i in p_data[:3]]) + f"\n |\n {(col_length - 6 if items else '|')}\n |\n" + "\n".join(["|"+i+"|" for i in p_data[-3:]])
        else:
            return "\n".join(["|"+i+"|" for i in p_data])

    def __str__(self):
        p_data, i_size = self._get_str_items()
        p_data = [" "*(i_size - len(p)) + p for p in p_data]
        if self.orientation == 'r':
            txt = self._format_row_str(p_data, row_length=self.length)
        else:
            txt = self._format_col_str(p_data, col_length=self.length)
        return txt
        
        
        

        
