from __future__ import annotations
from ._logiccore import LogicCore
from ._dvec import DVec
import operator, functools, itertools, copy

class DMatrix(LogicCore):
    """
    DMatrix is a dense matrix, all functionality uses native python.
    To initialize a matrix:

    DVec(data, [dtype=None])
    
    Parameters
    ----------
    data : 
        List((int, float, complex)),
            A single list will be a column vector.
        List(List((int, float, complex))),
            The inner list are used as row vectors.
        List(DVec),
            The DVecs are are used, where the orientation
            of the first vector is used.
        DVec,
            The orientation of the vector is used.

    dtype : {int, float, complex}, optional
        If this is given all items are cast to this type,
        else the first item of list is used as dtype.

    Returns
    -------
    X : DMatrix

    DVec is at its core a simple python list of DVecs, 
    the matrix allows some element wise unary operators:
        subtraction (-)
        muliplication (*)
        division (/)
        power (**)
        all (in)equalities (<, <=, ==, >=, >, !=)
        absolute (abs(x))
    If X and Y are Dmatrices then they need to be the same shape,
    If Y is a DVec its shape should match the axis where it is oriented.
    If Y is a scalar, it is performed element wise.

    Furthermore it supports the dot product with X @ Y.

    The matrix supports some fancy printing by print(DVec)
    """
    def __init__(self, data=None, dtype=None):
        self._format = "dmat"
        if data is None:
            return

        if isinstance(data, DVec):
            self._from_dvec(data)
            return

        elif not isinstance(data, list): 
            raise ValueError("Can only make a dense matrix from list or DVec")
        
        elif isinstance(data[0], DVec):
            self._from_dvecs(data)
            return

        else:
            self._from_data(data, dtype)
    
    def _from_data(self, data, dtype=None):
        # Column vector
        if not isinstance(data[0], list): 
            self.shape = (len(data), 1)
            self.data = DVec(data, dtype=dtype, orientation='c')
            return
        
        cols = len(data[0])
        rows = len(data)

        # Row vector
        if rows == 1:
            self.data = DVec(data[0], orientation='r')
            self.shape = (len(data[0]), 1)
            return

        # Col vector
        if cols == 1:
            self.data = DVec([d[0] for d in data], orientation='c')
            self.shape = (rows, 1)
            return
        
        # 2D vector
        self.data = [DVec(d, dtype=dtype, orientation='r') for d in data]
        self.shape = (rows, cols)
        if len(set([vec.dtype for vec in self.data])) != 1: raise ValueError(f"Inconsistent dtypes found: {set([vec.dtype.__name__ for vec in self.data])}")
        if any([vec.length != self.shape[1] for vec in self.data]): raise ValueError(f"""The following rows do not have length {self.shape[1]}: \n{", ".join([f'row {i} with length {vec.length}' for i, vec in enumerate(self.data) if vec.length != self.shape[1]])}""")

    def _from_dvec(self, dvec):
        if not isinstance(dvec, DVec): raise TypeError("Not DVec type")
        self.data = dvec
        if dvec.orientation == 'c':
            self.shape = (dvec.length, 1)
        else:
            self.shape = (1, dvec.length)

    def _from_dvecs(self, dvecs):
        if not isinstance(dvecs, list):
            raise TypeError("Can only initiate from a list of DVecs")
        if any([not isinstance(vec, DVec) for vec in dvecs]):
            raise TypeError("Not a list of DVecs")
        
        self.data = dvecs
        if len(set([vec.dtype for vec in self.data])) != 1: raise ValueError(f"Inconsistent dtypes found: {set([vec.dtype.__name__ for vec in self.data])}")
        if any([vec.length != self.data[0].length for vec in self.data]): raise ValueError(f"""The following DVecs do not have length {self.data[0].length}: \n{", ".join([f'row {i} with length {vec.length}' for i, vec in enumerate(self.data) if vec.length != self.data[0].length])}""")
        
        for vec in self.data:
            vec.orientation = self.orientation

        if self.orientation == 'r':
            self.shape = (len(self.data), self.data[0].length)
        else:
            self.shape = (self.data[0].length, len(self.data))

        self._fix_split_vector()
        
    @property
    def dtype(self):
        if isinstance(self.data, DVec):
            return self.data.dtype
        return self.data[0].dtype        

    @property
    def orientation(self):
        if isinstance(self.data, DVec):
            return self.data.orientation
        return self.data[0].orientation

    def index(self, a):
        """
        DMatrix.index(item)
        This will give the index of the item,

        Parameters:
        ----------
        a: item
            Item that needs to be indexed

        Returns:
        --------
        If the matrix is 2D:
            (x,y) with the index of the first match,
            note that the search patern depends on 
            the orientation, where the orientation
            is searched first.
        If the matrix is 1D:
            x the index of the first match
        """
        if isinstance(self.data, DVec):
            return self.data.index(a)

        for vec_i, vec in enumerate(self.data):
            try:
                vec_j = vec.index(a)
                if self.orientation == 'r':
                    return vec_i, vec_j
                return vec_j, vec_i
            except ValueError:
                continue
        raise ValueError(f"{a} not in matrix")

    def __getitem__(self, key) -> DMatrix:
        if isinstance(self.data, DVec):
            return self.data.__getitem__(key)

        self._force_orientation('r')
        key = self._parse_getkey(key)
        row_data = self._get_item_logic(key[0])
        if isinstance(row_data, DVec):
            slc_data = row_data[key[1]]
        else:
            slc_data = [row[key[1]] for row in row_data]

        if isinstance(slc_data, (int, float, complex)):
            return slc_data
        return DMatrix(data=slc_data)
    
    def __setitem__(self, key, val):
        key = self._parse_getkey(key)
        raise NotImplementedError
        
    def __add__(self, other) -> DMatrix:
        return self.__match_operator(other, operator.__add__)
    
    __iadd__ = __add__
    __radd__ = __add__
    
    def __sub__(self, other) -> DMatrix:
        return self.__match_operator(other, operator.__sub__)
    
    __isub__ = __sub__
    
    def __rsub__(self, other):
        return (-1 *self).__match_operator(other, operator.__add__)
    
    def __mul__(self, other) -> DMatrix:
        return self.__match_operator(other, operator.__mul__)
        
    __rmul__ = __mul__
    __imul__ = __mul__

    def __truediv__(self, other) -> DMatrix:
        return self.__match_operator(other, operator.__truediv__)
        
    def __rtruediv__(self, other) -> DMatrix:
        try:
            self = self ** -1
        except ZeroDivisionError:
            raise ZeroDivisionError("Can not divide by 0")
        return self.__match_operator(other, operator.__mul__)

    def __mod__(self, other) -> DMatrix:
        return self.__match_operator(other, operator.__mod__)
    
    def __rmod__(self, other):
        left = DMatrix(data = [[other] * self.shape[1]] * self.shape[0])
        
        return left.__match_operator(self, operator.__mod__)

    def __pow__(self, other) -> DMatrix:
        return self.__match_operator(other, operator.__pow__)
    
    def __rpow__(self, other):
        left = DMatrix(data = [[other] * self.shape[1]] * self.shape[0])
        return left.__match_operator(self, operator.__pow__)
        
    def __lt__(self, other) -> DMatrix:
        return self.__match_operator(other, operator.__lt__)
    
    def __le__(self, other) -> DMatrix:
        return self.__match_operator(other, operator.__le__)
    
    def __eq__(self, other) -> DMatrix:
        return self.__match_operator(other, operator.__eq__)
    
    def __ne__(self, other) -> DMatrix:
        return self.__match_operator(other, operator.__ne__)
    
    def __ge__(self, other) -> DMatrix:
        return self.__match_operator(other, operator.__ge__)
    
    def __gt__(self, other) -> DMatrix:
        return self.__match_operator(other, operator.__gt__)
    
    __rlt__ = __ge__
    __rle__ = __gt__
    __req__ = __eq__
    __rne__ = __ne__
    __rge__ = __lt__
    __rgt__ = __le__
    
    def __abs__(self) -> DMatrix:
        return self.__self_operator(operator.__abs__)
    
    def __match_operator(self, other, opp):
        if isinstance(self.data, DVec) and hasattr(other, '_format') and isinstance(other.data, DVec):
            opp_data = opp(self.data, other.data)

        elif isinstance(self.data, DVec):
            opp_data = opp(self.data, other)

        elif isinstance(other, DMatrix):
            if self.shape != other.shape:
                raise ValueError(f"Matrices are not the same size: {self.shape} and {other.shape}")
            other._force_orientation(orientation=self.orientation)
            opp_data = [opp(s_vec, o_vec) for s_vec, o_vec in zip(self.data, other.data)]

        elif isinstance(other, (int, float, complex)):
            opp_data = [opp(vec, other) for vec in self.data]

        elif isinstance(other, DVec):
            opp_data = [opp(vec, other) for vec in self.data]

        elif hasattr(other, "_format"):
            return NotImplemented
        
        else:
            return NotImplementedError
        
        return DMatrix(data=opp_data)
        
    def __self_operator(self, opp) -> DMatrix:
        if isinstance(self.data, DVec):
            opp_data = opp(self.data)
        else:
            opp_data = [opp(vec) for vec in self.data]
        return DMatrix(data=opp_data)

    def __matmul_self(self, other) -> DMatrix:
        self._force_orientation('r')
        other._force_orientation('c')

        if isinstance(other.data, DVec) and isinstance(self.data, DVec):
            dot_data = self.data @ other.data
        elif isinstance(other.data, DVec):
            dot_data = [row @ other.data for row in self.data]
        elif isinstance(self.data, DVec):
            dot_data = [[self.data @ col] for col in other.data]
        else:
            dot_data = [[row @ col for col in other.data] for row in self.data]
        return dot_data
    
    def __matmul__(self, other) -> DMatrix:
        if not hasattr(other, "_format"): return NotImplemented
        if not isinstance(other, DVec) and self.shape[1] != other.shape[0] : raise ValueError(f"Can not do a dot product with between matrices with size {self.shape} and {other.shape}")
        if isinstance(other, DMatrix):
            dot_data = self.__matmul_self(other)
        elif isinstance(other, DVec):
            dot_data = self.__matmul_self(DMatrix(other))
        else:
            return NotImplemented
            
        if isinstance(dot_data, list):
            return DMatrix(data=dot_data)
        return dot_data
    
    def __rmatmul__(self, other):
        return (self.T.__matmul__(other.T)).T
    
    @property
    def T(self) -> DMatrix:
        """
        DMatrix.T

        Returns a tranposed copy of itself.
        """
        new = copy.deepcopy(self)
        if isinstance(new.data, DVec):
            new.data = new.data.T
        else:
            new.data = [vec.T for vec in new.data]
        new.shape = (new.shape[1], new.shape[0])
        return new
    
    def _T_inplace(self):
        if isinstance(self.data, DVec):
            self.data = self.data.T
        else:
            self.data = [vec.T for vec in self.data]
        self.shape = (self.shape[1], self.shape[0])
    
    def _force_orientation(self, orientation):
        if self.orientation == orientation: 
            return
        if isinstance(self.data, DVec):
            f_data = [[d] for d in self.data]
        else:
            f_data = [[vec[i] for vec in self.data] for i in range(self.data[0].length)]
        if len(f_data) == 1:
            self.data = DVec(f_data[0], dtype=self.dtype, orientation=orientation)
        else:
            self.data = [DVec(f, dtype=self.dtype, orientation=orientation) for f in f_data]

    def _fix_split_vector(self):
        if not 1 in self.shape:
            return
        if isinstance(self.data, DVec):
            return
        
        if self.shape[0]==1:
            self._force_orientation('r')
        else:
            self._force_orientation('c')
        
        if isinstance(self.data, list):
            self.data - self.data[0]

    def reshape(self, shape, order='R') -> DMatrix:
        """
        DMatrix.reshape(shape, order='R')

        Reshapes the matrix into the shape given shape.

        Parameters:
        -----------
        shape: tuple of ints
            The new shape of the matrix, the amount of elements
            of the old matrix need to match the amount in the new
            matrix.
        order: {'C', 'R'}
            The order which the elements are read from the old matrix.
            'R' denotes row by row, and 'C' column by column.

        Returns
        A reshaped DMatrix, the old matrix is consumed in the process.
        """
        if functools.reduce(operator.__mul__, shape) != functools.reduce(operator.__mul__, self.shape): raise ValueError(f"Cannot cast ({self.shape}) into {shape}")
        if not isinstance(self.data, DVec):
            self.flatten()

        if self.orientation != 'r':
            self._T_inplace()

        self.data = [self.data[s:s+shape[1]] for s in range(0, self.shape[1], shape[1])]
        self.shape = shape
        return self

    def flatten(self, order="R") -> DMatrix:
        """
        DMatrix.flatten(order='R')

        flattens the matrix.

        Parameters:
        -----------
        order: {'C', 'R'}
            The order which the elements are read from the old matrix.
            'R' denotes row by row, and 'C' column by column.

        Returns
        A flattend DMatrix
        """
        if order=="R":
            self._force_orientation('r')
        elif order=="C":
            self._force_orientation('c')
        else:
            raise ValueError("Order not valid")
        self.data = DVec(list(itertools.chain(*self.data)), orientation='c')
        self.shape = (1, self.data.length)
        return self

    def round(self, r=2) -> DMatrix:
        """
        DMatrix.round(r):
        Rounds all elements in data to r digits.
        """
        if isinstance(self.data, DVec):
            r_data = self.data.round(2)
        else:
            r_data = [row.round(r) for row in self.data]
        return DMatrix(data=r_data)

    def sum(self, axis=None):
        """
        DMatrix.sum(axis=None)

        This will sum the matrix along an axis if given.
        Else the sum of all elements if axis is None.
        """
        if isinstance(self.data, DVec):
            return sum(self.data)
        elif axis is None:
            return sum([sum(vec) for vec in self.data])
        elif axis == 0:
            self._force_orientation('r')
            return DMatrix(sum(self.data))
        else:
            self._force_orientation('c')
            return DMatrix(sum(self.data))
        
    def tolist(self):
        if not isinstance(self.data, DVec):
            self._force_orientation('r')
            return [row_vec.tolist() for row_vec in self.data]
        else:
            return self.data.tolist()
    
    @classmethod
    def eye(cls, shape):
        if isinstance(shape, tuple) and all([isinstance(int, s) for s in shape]):
            eye_data = [[0] * i + [1] + [0] * (shape[1] - i - 1) for i in range(shape[0])]
        elif isinstance(shape, int):
            eye_data = [[0] * i + [1] + [0] * (shape - i - 1) for i in range(shape)]
        else:
            raise ValueError("Shape needs to be an integer or a tuple of integer")
        return DMatrix(eye_data)
    
    @classmethod
    def arange(cls, *args):
        """
        DMatrix.arange([start], stop, [step])
        
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
        X : DMatrix
            A column vector with items in the given range.
        """
        return DMatrix(DVec.arange(*args))
        
    def __str__(self):
        if isinstance(self.data, DVec):
            return self.data.__str__()
        self._force_orientation('r')
        if isinstance(self.data, DVec):
            return self.data.__str__()
        
        p_data = [vec._get_str_items() for vec in (self.data if len(self.data) < 6 else self.data[:3] + self.data[-3:])]
        i_size = max([p[1] for p in p_data])
        p_data =  [[" "*(i_size - len(item)) + item for item in line] for line in [p[0] for p in p_data]]
        row_txt = [DVec._format_row_str(p, row_length=self.shape[1], items=i==0) for i, p in enumerate(p_data)]
        if self.shape[0] > 6:
            v_lines_left = " "*i_size + "|" + "|".join([" " * (i_size + 1)] * min(3, self.shape[1]//2))
            v_lines_right = "|" + "|".join([" " * (i_size + 1)] * min(3, (self.shape[1] - 1)//2  + 1))
            gap_txt = v_lines_left + ("" if self.shape[1]<6 else " "*5) + v_lines_right + "\n"
            txt = "\n".join(row_txt[:3])+ "\n" + gap_txt + gap_txt[:i_size] + f"{self.shape[0] - 6}" + gap_txt[i_size + 1:]  + gap_txt + "\n".join(row_txt[-3:])
        else: 
            txt = "\n".join(row_txt)
        
        return txt