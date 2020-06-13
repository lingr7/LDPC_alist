import numpy as np


# 校验矩阵的读取
def alistToNumpy(lines):
    """Converts a parity-check matrix in AList format to a 0/1 numpy array. The argument is a
    list-of-lists corresponding to the lines of the AList format, already parsed to integers
    if read from a text file.
    The AList format is introduced on http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html.
    This method supports a "reduced" AList format where lines 3 and 4 (containing column and row
    weights, respectively) and the row-based information (last part of the Alist file) are omitted.
    Example:
        alistToNumpy([[3,2], [2, 2], [1,1,2], [2,2], [1], [2], [1,2], [1,2,3,4]])
        array([[1, 0, 1],
               [0, 1, 1]])
    """
    nCols, nRows = lines[0]
    if len(lines[2]) == nCols and len(lines[3]) == nRows:
        startIndex = 4
    else:
        startIndex = 2
    matrix = np.zeros((nRows, nCols), dtype=np.int)
    for col, nonzeros in enumerate(lines[startIndex:startIndex + nCols]):
        for rowIndex in nonzeros:
            if rowIndex != 0:
                matrix[rowIndex - 1, col] = 1
    return matrix

def save_alist(name, mat, j=None, k=None):

    H=np.copy(mat)
    # H=H.T

    '''
    Function converts parity check matrix into the format required for the LDPC decoder
    '''

    if j is None:
        j=int(max(H.sum(axis=0)))


    if k is None:
        k=int(max(H.sum(axis=1)))


    m, n = H.shape # rows, cols
    f = open(name, 'w')
    print(n, m, file=f)
    print(j, k, file=f)

    for col in range(n):
        print( int(H[:, col].sum()), end=" ", file=f)
    print(file=f)
    for row in range(m):
        print( int(H[row, :].sum()), end=" ", file=f)
    print(file=f)

    for col in range(n):
        for row in range(m):
            if H[row, col]:
                print( row+1, end=" ", file=f)
        print(file=f)

    for row in range(m):
        for col in range(n):
            if H[row, col]:
                print(col+1, end=" ", file=f)
        print(file=f)
    f.close()
    

def alist2sparse(file_H):
    with open(file_H) as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            new_lines.append(list(map(int, line.split())))
    H_matrix = alistToNumpy(new_lines)
    return H_matrix

def alist2sparse2(fname):
    a = np.loadtxt(fname, delimiter='\n',dtype=str)
    alist_n = a.shape[0]
    list_a = []
    for i in range(alist_n-1):
        k = a[i].split()
        #print(k)
        list_a.extend(k)
    a=np.array(list_a,dtype=np.int32)
    #处理掉补位的无效的0索引
   
    #Read file contents as an array  
    max_index = a.shape[0]
    mat_n = a[0]
    mat_m = a[1]
    maxincol = a[2]
    # print(maxincol)
    maxinrow = a[3]
    num = sum(a[4:4+mat_n])#非0元素个数
    index_col_num = a[4:4+mat_n]#逐列的非0元素个数
    # print(num)
    start = 4 + mat_m + mat_n
    b=a[a>0]
    k = 0
    H = np.zeros((mat_m,mat_n),dtype=np.int32)
    for i in range(mat_n):#共有7列，每一列的索引都要读进来,重建索引
        for j in range(index_col_num[i]):#出现的最多列数，不代表所有列。
            if(k==(num)):
                break
            H[b[start+k]-1,i]=1
            k = k+1
    #没有处理补位的无效的0索引
    return H 

def save_mat_txt(tname,H):
    a=np.nonzero(H==1)#一个记录了行索引，一个记录了列索引，这里换成np.where(H)也可以吧。
    #b,c=np.where(a)#具有类似的效果 
    # c = np.where(H[0,:])
    # c = np.array(c,np.int32)
    c= np.array(np.where(H),np.int32)
    print(c.shape)
    #元组中保存了一个数组，怎么处理成数组，我的尝试失败了。
    # b=np.nonzero(H[0,:])
    # print(b)
    num = len(a[0])
    f = open(tname, 'w')
    for i in range(num):
        print(a[0][i]," ",a[1][i],file=f)
    f.close()
    
    
def test_save_alist():
    
    hamming_matrix=np.array([[1,0,0,1,1,0,1],
                             [0,1,0,1,0,1,1],
                             [0,0,1,0,1,1,1]])

    save_alist("hamming_d_3.alist",hamming_matrix)

def test_2_alist2sparse():
    H = alist2sparse('CCSDS_ldpc_n256_k128.alist')
    #save_alist("55.alist", H)
    save_mat_txt('55.txt',H)
    H_2 = alist2sparse2('CCSDS_ldpc_n256_k128.alist')

    #print((H==H_2).all())



 
    
# H_m = alist2sparse2('CCSDS_ldpc_n128_k64.alist')
# #为什么有时候是元组，有时候是数组呢？
# #是元组的时候怎么办？是数组的时候怎么办？
# # print(H_m.type)
# print(H_m.shape)
# tname = 'LDPC_chk_mat_128_64.txt'
# save_mat_txt(H_m,tname)


# print(H_m.shape())
# if __name__=="__main__":
#     main()
    #test_save_alist()
    # test_2_alist2sparse()
    