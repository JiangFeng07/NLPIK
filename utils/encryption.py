#!/usr/local/bin/ python3
# -*- coding: utf-8 -*-
# @Time : 2022-03-01 17:17
# @Author : Leo


# -*- coding: utf-8 -*-
import base64
from Crypto.Cipher import AES

AES_SECRET_KEY = 'a' * 32  # 此处16|24|32个字符
IV = "1234567890123456"

# padding算法
BS = len(AES_SECRET_KEY)
pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS)
unpad = lambda s: s[0:-ord(s[-1:])]


class AES_ENCRYPT(object):
    # 加密函数
    @staticmethod
    def encrypt(text):
        cryptor = AES.new(AES_SECRET_KEY.encode("utf8"), AES.MODE_ECB, IV.encode("utf8"))
        ciphertext = cryptor.encrypt(bytes(pad(text), encoding="utf8"))
        # AES加密时候得到的字符串不一定是ascii字符集的，输出到终端或者保存时候可能存在问题，使用base64编码
        return base64.b64encode(ciphertext)

    # 解密函数
    def decrypt(text):
        decode = base64.b64decode(text)
        cryptor = AES.new(AES_SECRET_KEY.encode("utf8"), AES.MODE_ECB, IV.encode("utf8"))
        plain_text = cryptor.decrypt(decode)
        return unpad(plain_text)


if __name__ == '__main__':
    my_email = "1111111111"
    e = AES_ENCRYPT.encrypt(my_email)
    d = AES_ENCRYPT.decrypt(e)
    # print(my_email)
    print(type(e.decode('utf-8')))
    print(d.decode('utf-8'))
