from __future__ import annotations
import operator, functools, time, itertools

class PMatrix:
    def __init__(self, data=None, shape=1, fill=0):
        if data:
            self.overwrite_data(data)
        elif isinstance(shape, int) or len(shape) == 1:
            self.shape = ((shape[0], 1) if isinstance(shape, tuple) else (shape, 1))
            self.data = [fill] * self.shape[0]
        elif isinstance(shape, (tuple, list)):
            self.shape = tuple(shape[:2])
            self.data = [[fill] * shape[1]] * shape[0]

    def overwrite_data(self, data):
        rows = len(data)
        if isinstance(data[0], list):
            cols = len(data[0])
            if any([len(row) != cols for row in data]): raise ValueError("Not all rows are the same dimension")
            if any([hasattr(i, '__iter__') for row in data for i in row]): raise ValueError("Cannot nest data 3 deep")
        else:
            cols = 1

        if rows == 1:
            data = data[0]
        if cols == 1 and isinstance(data[0], list):
            data = [d[0] for d in data]

        self.data = data
        self.shape = (rows, cols)
    
    def index(self, a):
        if 1 in self.shape:
            return self.data.index(a)

        for row_i, row in enumerate(self.data):
            try:
                col_i = row.index(a)
                return (row_i, col_i)
            except ValueError:
                continue
        raise ValueError(f"{a} not in matrix")

    def __getitem__(self, key) -> PMatrix:
        if 1 in self.shape:
            if isinstance(key, (list, tuple)):
                return operator.itemgetter(*key)(self.data)
            return self.data[key]

        if isinstance(key, (int, slice)):
            key = (key, )
        elif isinstance(key, list):
            key = (key, )

        data = self.__data_slicing(self.data, key)
        if len(data) == 1 and len(data[0]) == 1:
            return data[0][0]
        return PMatrix(data=data)
            
    def __data_slicing(self, data, key, i=0):
        if isinstance(key[0], list):
            data = operator.itemgetter(*key[0])(data)
            if len(key[0]) == 1:
                data = [data]

        elif isinstance(key[0], slice):
            data = operator.itemgetter(key[0])(data)
        elif isinstance(key[0], int):
            data = [data[key[0]]]
        else:
            raise ValueError(f"Can not index using {key[0]}, must be tuple, list or int")
        
        if len(key) == 1:
            return data
        else:
            return [self.__data_slicing(d, key[1:], i + 1) for d in data]
        
    def __add__(self, other) -> PMatrix:
        return self.__match_operator(other, operator.__add__)
    
    __radd__ = __add__
    
    def __sub__(self, other) -> PMatrix:
        return self.__match_operator(other, operator.__sub__)
    
    __rsub__ = __sub__
    
    def __mul__(self, other) -> PMatrix:
        return self.__match_operator(other, operator.__mul__)
        
    __rmul__ = __mul__

    def __truediv__(self, other) -> PMatrix:
        return self.__match_operator(other, operator.__truediv__)
        
    __rtruediv__ = __truediv__

    def __mod__(self, other) -> PMatrix:
        return self.__match_operator(other, operator.__mod__)

    def __pow__(self, other) -> PMatrix:
        return self.__match_operator(other, operator.__pow__)
    
    def __lt__(self, other) -> PMatrix:
        return self.__match_operator(other, operator.__lt__)
    
    def __le__(self, other) -> PMatrix:
        return self.__match_operator(other, operator.__le__)
    
    def __eq__(self, other) -> PMatrix:
        return self.__match_operator(other, operator.__eq__)
    
    def __ne__(self, other) -> PMatrix:
        return self.__match_operator(other, operator.__ne__)
    
    def __ge__(self, other) -> PMatrix:
        return self.__match_operator(other, operator.__ge__)
    
    def __gt__(self, other) -> PMatrix:
        return self.__match_operator(other, operator.__gt__)
    
    def __abs__(self) -> PMatrix:
        return self.__self_operator(operator.__abs__)
    
        
    def __match_operator(self, other, opp) -> PMatrix:
        if isinstance(other, PMatrix):
            if self.shape == other.shape:
                opp_data = [[opp(i,j) for i, j in zip(row_s, row_o)] for row_s, row_o in zip(self.data, other.data)]
            
            elif self.shape[0] == other.shape[0] and other.shape[1] == 1:
                opp_data = [[opp(i, scalar) for i in row] for row, scalar in zip(self.data, other.data)]

            elif self.shape[0] == other.shape[0] and self.shape[1] == 1:
                opp_data = [[opp(scalar, i) for i in row] for scalar, row in zip(self.data, other.data)]

            elif self.shape[1] == other.shape[1] and other.shape[0] == 1:
                opp_data = [[opp(i,j) for i, j in zip(row, other.data)] for row in self.data]
            
            elif self.shape[1] == other.shape[1] and self.shape[0] == 1:
                opp_data = [[opp(i,j) for i, j in zip(self.data, row)] for row in other.data]
            else:
                raise ValueError(f"Matrix with size ({self.rows},{self.cols}) can not be aligned with ({other.rows},{other.cols})")
        
        else:
            if 1 in self.shape:
                opp_data = [opp(i, other) for i in self.data]
            else:
                opp_data = [[opp(i,other) for i in row] for row in self.data]

        return PMatrix(data=opp_data)
    
    def __self_operator(self, opp) -> PMatrix:
        if 1 in self.shape:
            opp_data = list(map(opp, self.data))
        else:
            opp_data = [list(map(opp, row)) for row in self.data]
        return PMatrix(data=opp_data)

    def __matmul__(self, other) -> PMatrix:
        if not isinstance(other, PMatrix):
            return self * other
        elif other.shape[0] * other.shape[1] == 1:
            return self * other.data[0]

        if self.shape[1] != other.shape[0]: 
            raise ValueError(f"Cannot do a dot product on {self.shape}, {other.shape}")

        if other.shape[1] == 1:
            dot_data = [sum([l*r for l,r in zip(row, other.data)]) for row in self.data]
        elif self.shape[0] == 1:
            other.T(inplace=True)
            dot_data = [sum([l*r for l,r in zip(self.data, col)]) for col in other.data]
            other.T(inplace=True)
        elif self.shape[1] == 1 and other.shape[0] == 1:
            dot_data = [[i * j for j in other.data] for i in self.data]
        else:
            other.T(inplace=True)
            dot_data = [sum([l*r for l, r in zip(self.data[i], other.data[j])]) for (i, j) in itertools.product(range(self.shape[0]), range(other.shape[0]))]
            dot_data = [dot_data[s:s+other.shape[0]] for s in range(0, len(dot_data), other.shape[0])]
            other.T(inplace=True)
        return PMatrix(data=dot_data)
    
    def T(self, inplace=False) -> PMatrix:
        if 1 in self.shape and not inplace:
            self_t = PMatrix(data=self.data)
            self_t.shape = (self_t.shape[1], self_t.shape[0])
            return self_t
        
        if 1 in self.shape:
            self.shape = (self.shape[1], self.shape[0])
            return
        
        t_data = [[row[i] for row in self.data] for i in range(self.shape[1])]

        if inplace:
            self.shape = (self.shape[1], self.shape[0])
            self.data = t_data
            return
            
        return PMatrix(data=t_data)
    
    def reshape(self, shape) -> PMatrix:
        if functools.reduce(operator.__mul__, shape) != functools.reduce(operator.__mul__, self.shape): raise ValueError(f"Cannot cast ({self.shape}) into {shape}")
        if self.shape[0] != 1 and self.shape[1] == 1: 
            self.T(inplace=True)
        elif self.shape[0] !=1:
            self = self.flatten().T(inplace=True)

        self.data = [self.data[s:s+shape[1]] for s in range(0, self.shape[1], shape[1])]
        self.shape = shape
        return self

    def flatten(self, order="R") -> PMatrix:
        if order=="R":
            self.data = list(itertools.chain(self.data))
        elif order=="C":
            self.data = list(itertools.chain(self.T(inplace=True).data))
        else:
            raise ValueError("Order not valid")
        
        self.rows = 1
        self.cols = len(self.data)
        return self

    def round(self, r=2) -> PMatrix:
        r_data = [[round(i,r) for i in row] for row in self.data]
        return PMatrix(data=r_data)

    def sum(self, axis=None):
        if 1 in self.shape:
            return sum(self.data)
        elif axis == 0: 
            return PMatrix(data=[sum(row) for row in self.data])
        elif axis == 1: 
            return PMatrix(data=[sum([row[i] for row in self.data]) for i in range(self.shape[1])]).T(inplace=True)
        else:
            return sum([sum(row) for row in self.data])
        
    def __str__(self):
        def txt_row(row, i_size, length, items=True):
            if length > 8:
                txt = "[" + ", ".join(row[:3]) + f" --{(length - 6 if items else '-' * len(str(length-6)))}-- " + ", ".join(row[-3:]) + "]"
            else:
                txt = "[" + ", ".join(row) + "]"
            return txt
        
        match self.shape:
            case (1, x) | (x, 1) if x > 8:
                p_data = self.data[:3] + self.data[-3:]
            case (1, x) | (x, 1):
                p_data = self.data
            case (x, y) if x > 8 and y > 8:
                p_data = itertools.chain(*([row[:3] + row[-3:] for row in self.data[:3]] + [row[:3] + row[-3:] for row in self.data[-3:]]))
            case (x,y) if y>8:
                p_data = itertools.chain(*[row[:3] + row[-3:] for row in self.data])
            case (x,y) if x>8:
                p_data = itertools.chain(*(self.data[:3] + self.data[-3:]))
            case (x,y):
                p_data = itertools.chain(*self.data)
                
        p_data = list(map(str, p_data))
        i_size = max(map(len, p_data))
        p_data = [" "*(i_size - len(p)) + p for p in p_data]

        match self.shape:
            case (1, x):
                txt = txt_row(p_data, i_size=i_size, length=x)
            case (x, 1) if x > 8:
                txt = "\n".join(["|"+i+"|" for i in p_data[:3]]) + f"\n |\n {x - 6}\n |\n" + "\n".join(["|"+i+"|" for i in p_data[-3:]])
            case (x, 1):
                txt = "\n".join(["|"+i+"|" for i in p_data])         
            case (x, y) if x>8:
                step = min(y, 6)
                txt = "[" + "\n ".join([txt_row(p_data[c:c+step], i_size=i_size, length=y, items=i==0) for i, c in enumerate(range(0, len(p_data)//2, step))]) + f"\n |\n |{x-6}\n |\n "+ "\n ".join([txt_row(p_data[c:c+step], i_size=i_size, length=y, items=False) for i, c in enumerate(range(len(p_data)//2, len(p_data), step))]) + "]"
            case (x,y):
                step = min(y, 6)
                txt = "[" + "\n ".join([txt_row(p_data[c:c+step], i_size=i_size, length=y, items=i==0) for i, c in enumerate(range(0, len(p_data), step))]) + "]"
            # case (x,y):
                # txt = "[" + "\n ".join([txt_row(p_data[c:c+6], i_size=i_size, length=y, items=i==0) for i, c in enumerate(range(0, len(p_data), 6))]) + "]"

        return txt

    @classmethod
    def eye(cls, n, cons=1, diag=0):
        if abs(diag) >= n: raise ValueError("Diagonal out of bounds of size")
        data = [0] * n * n
        for i in filter(lambda x: x>=0 and x < n * n, range(-diag * n, (n - diag) * n, n + 1)):
            data[i] = cons
        return cls(data=data).reshape((n,n))
    
    @classmethod
    def u_t(cls, n, cons=1):
        return sum([PMatrix().eye(n, diag=i) for i in range(n)])
    

if __name__=="__main__":
    t1 = time.time()
    X = PMatrix(data=list(range(100000))).reshape((10000,10))
    # Y = PMatrix(data=list(range(10))).reshape((5,2))
    # print(X.T() @ X)
    X @ X.T()
    # print(X)
    print(time.time() - t1)