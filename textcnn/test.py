'''
Dropoou层demo
'''
import tensorflow as tf
dropout = tf.placeholder(tf.float32)#设定一个dropout参数
x = tf.Variable(tf.ones([10, 10]))
y = tf.nn.dropout(x, dropout)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
out1 = sess.run(y, feed_dict = {dropout: 0.5})#有0.5的神经元是未被激活的
print(out1)
'''
tf.where用法demo
'''
import tensorflow as tf
a = [[1,2,3],[4,5,6]]
b = [[1,0,3],[1,5,1]]
condition1 = [[True,False,False],
             [False,True,True]]
condition2 = [[True,False,False],
             [False,True,False]]

print(sess.run(tf.where(condition1)))
print(sess.run(tf.where(condition2)))