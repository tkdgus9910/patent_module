# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:56:09 2023

@author: tmlab
"""

import multiprocessing

def worker(num):
    
    """작업자 함수"""
    result = num * 2
    print(f'Worker: {result}')

if __name__ == '__main__':
    # CPU 코어의 개수를 가져옵니다
    num_cores = multiprocessing.cpu_count()

    # 작업을 위한 프로세스 풀을 생성합니다
    pool = multiprocessing.Pool(processes=num_cores-1)

    # 작업을 프로세스 풀에 매핑합니다
    numbers = [1, 2, 3, 4, 5]
    pool.map(worker, numbers)

    # 프로세스 풀을 닫고 종료합니다
    pool.close()
    pool.join()