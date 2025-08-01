import weakref
import numpy as np
import contextlib
import package

# =============================================================================
# Config
# =============================================================================

class Config:
    enable_backprop = True # 逆伝播有効モード
    train = True # 学習モード

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

def test_mode():
    return using_config('train', False)

# =============================================================================
# Variable / Function
# =============================================================================

class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        # ndarray以外の型を入力したときのエラーメッセージ
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        
        # 変数という箱の中の変数要素
        self.data = data
        self.name = name
        self.grad = None
        self.generation = 0
        # 変数という箱の中の関数要素
        self.creator = None

    def __len__(self): # # Variableクラスに対して len を使えば、__len__メソッドが代わりに呼ばれる。
        return len(self.data)
    
    def __repr__(self): # Variableクラスに対して print を使えば、__repr__メソッドが代わりに呼ばれる。
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def unchain(self):
        self.creator = None

    def cleargrad(self): # 同じ変数を使って複数回違う計算を行う際に使う初期化メソッド
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data)) # dataと同じ大きさの全要素1の行列を作成

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop() # 関数を取得
            gys = [output().grad for output in f.outputs] # funcごとの出力側のgradをリスト化、()はweakrefの影響

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys) # リスト化されたgradをアンパックして、funcのbackwardからfuncの入力側のgradを得る
                if not isinstance(gxs, tuple):
                    gxs = (gxs,) # funcの入力側のgradをタプル化
            
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None: # funcの入力側のgradに何もなければ、そのまま
                        x.grad = gx
                    else: # funcの入力側のgradが既に存在していれば、 足し合わせる(同じ変数が足されたときにこの事案発生)
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)
        
            if not retain_grad: # gradをすべて保持するかどうか
                for y in f.outputs:
                    y().grad = None # yはweakref
    
    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return package.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return package.functions.transpose(self, axes)
    
    @property
    def T(self):
        return package.functions.transpose(self)
    
    def sum(self, axis=None, keepdims=False):
        return package.functions.sum(self, axis, keepdims)

class Parameter(Variable): # Variableクラスの継承
    pass

# np.ndarray以外の数字の型をnp.ndarrayに変換する便利関数 (Numpyの仕様上入れないと仕方ない)
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# Variableのインスタンスでないもの(ndarray)をVariableのインスタンス化するコード
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Function:
    def __call__(self, *inputs): # *をつけることで可変長引数を与えて関数を呼ぶことができる
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs] # 入力をdataとして保存し、リスト化
        ys = self.forward(*xs) # *をつけてアンパッキング　例) [x0, x1] の場合、self.forward(x0, x1)としてアンパックされる
        if not isinstance(ys, tuple): # タプルではない場合、タプルにする
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] # output = Variable(as_array(y)) # Variableとして返す

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self) # 出力変数に生みの親を覚えさせる
            self.inputs = inputs # 入力された変数を覚える
            self.outputs = [weakref.ref(output) for output in outputs] # 出力も覚える(メモリ容量圧迫対策でweakrefを使用)
            
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()
    
# =============================================================================
# 四則演算 / 演算子のオーバーロード
# =============================================================================

class Add(Function): # AddクラスはFunctionクラスを継承
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # for broadcaset
            gx0 = package.functions.sum_to(gx0, self.x0_shape)
            gx1 = package.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

class Mul(Function): # MulクラスはFunctionクラスを継承
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = package.functions.sum_to(gx0, x0.shape)
            gx1 = package.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

class Neg(Function): # NegクラスはFunctionクラスを継承
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = package.functions.sum_to(gx0, self.x0_shape)
            gx1 = package.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

class Div(Function): # DivクラスはFunctionクラスを継承
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = package.functions.sum_to(gx0, x0.shape)
            gx1 = package.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return div(x1, x0)

class Pow(Function): # PowクラスはFunctionクラスを継承
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

def setup_variable(): # 基本演算の設定
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = package.functions.get_item