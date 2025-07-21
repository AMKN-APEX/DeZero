import numpy as np
import unittest

class Variable:
    def __init__(self, data):
        # ndarray以外の型を入力したときのエラーメッセージ
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    # 再帰的アプローチ
    # def backward(self):
    #     f = self.creator # 1.関数を取得
    #     if f is not None:
    #         x = f.input # 2.関数の入力を取得
    #         x.grad = f.backward(self.grad) # 3.関数のbackwardメソッドを呼ぶ
    #         x.backward() # 自分よりも1つ前の変数のbackwardメソッドを呼ぶ(再帰)

    # forループアプローチ
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data) # dataと同じ大きさの全要素1の行列を作成

        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 関数を取得
            x, y = f.input, f.output # 関数の入出力を取得
            x.grad = f.backward(y.grad) #   backwardメソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator) # 1つ前の関数をリストに追加

class Function:
    def __call__(self, input):
        x = input.data # データを取り出す
        y = self.forward(x) # 具体的なことはforwardメソッドで行う
        output = Variable(as_array(y)) # Variableとして返す
        output.set_creator(self) # 出力変数に生みの親を覚えさせる
        self.input = input # 入力された変数を覚える
        self.output = output # 出力も覚える
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad) # 2値がほとんど同じ値ならTrueを出す関数
        self.assertTrue(flg)


# np.ndarray以外の数字の型をnp.ndarrayに変換する便利関数 (Numpyの仕様上入れないと仕方ない)
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# 中心差分近似
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

# squareのdef(3行)
# def square(x):
#     f = Square()
#     return f(x)

# expのdef(3行)
# def exp(x):
#     f = Exp()
#     return f(x)

# squareのdef(2行)
def square(x):
    return Square()(x) # 1行でまとめて書く

# expのdef(2行)
def exp(x):
    return Exp()(x) # 1行でまとめて書く

####################実行####################
x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)

