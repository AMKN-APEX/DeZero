import numpy as np
import unittest
import weakref

class Variable:
    def __init__(self, data):
        # ndarray以外の型を入力したときのエラーメッセージ
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        
        # 変数という箱の中の変数要素
        self.data = data
        self.grad = None
        self.generation = 0
        # 変数という箱の中の関数要素
        self.creator = None

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self): # 同じ変数を使って複数回違う計算を行う際に使う初期化メソッド
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data) # dataと同じ大きさの全要素1の行列を作成

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

# np.ndarray以外の数字の型をnp.ndarrayに変換する便利関数 (Numpyの仕様上入れないと仕方ない)
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs): # *をつけることで可変長引数を与えて関数を呼ぶことができる
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
    
class Config:
    enable_backprop = True # 逆伝播有効モード
    
class Square(Function): # SquareクラスはFunctionクラスを継承
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data # x = self.input.data
        gx = 2 * x * gy
        return gx

# squareのdef(2行)
def square(x):
    return Square()(x) # 1行でまとめて書く
    
class Exp(Function): # ExpクラスはFunctionクラスを継承
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data # x = self.input.data
        gx = np.exp(x) * gy
        return gx

# expのdef(2行)
def exp(x):
    return Exp()(x) # 1行でまとめて書く
    
class Add(Function): # AddクラスはFunctionクラスを継承
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)
    
# class SquareTest(unittest.TestCase):
#     def test_forward(self):
#         x = Variable(np.array(2.0))
#         y = square(x)
#         expected = np.array(4.0)
#         self.assertEqual(y.data, expected)

#     def test_backward(self):
#         x = Variable(np.array(3.0))
#         y = square(x)
#         y.backward()
#         expected = np.array(6.0)
#         self.assertEqual(x.grad, expected)

#     def test_gradient_check(self):
#         x = Variable(np.random.rand(1))
#         y = square(x)
#         y.backward()
#         num_grad = numerical_diff(square, x)
#         flg = np.allclose(x.grad, num_grad) # 2値がほとんど同じ値ならTrueを出す関数
#         self.assertTrue(flg)

# 中心差分近似
# def numerical_diff(f, x, eps=1e-4):
#     x0 = Variable(x.data - eps)
#     x1 = Variable(x.data + eps)
#     y0 = f(x0)
#     y1 = f(x1)
#     return (y1.data - y0.data) / (2 * eps)

####################実行####################
# x = Variable(np.array(0.5))
# y = square(exp(square(x)))
# y.backward()
# print(x.grad)

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
t = add(square(x0), square(x1))
y = add(t, x1)
y.backward()
print(y.grad, t.grad)
print(x0.grad, x1.grad)

