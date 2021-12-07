import random
import qrcode


def run():
    for i in range(5):
        randID = random.randint(1, 50000)
        img = qrcode.make(randID, version=1, box_size=12,
                          error_correction=qrcode.constants.ERROR_CORRECT_H).convert('RGB')
        img.save('./out/{}_H.png'.format(randID))
        img = qrcode.make(randID, version=1, box_size=12,
                          error_correction=qrcode.constants.ERROR_CORRECT_L).convert('RGB')
        img.save('./out/{}_L.png'.format(randID))


if __name__ == '__main__':
    run()
