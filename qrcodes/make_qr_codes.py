import random
import qrcode


def run():
    for i in range(5):
        rand_id = random.randint(1, 50000)
        img = qrcode.make(str(rand_id) + '_H', version=1, box_size=12,
                          error_correction=qrcode.constants.ERROR_CORRECT_H).convert('RGB')
        img.save('./out/{}_H.png'.format(rand_id))
        img = qrcode.make(str(rand_id) + '_L', version=1, box_size=12,
                          error_correction=qrcode.constants.ERROR_CORRECT_L).convert('RGB')
        img.save('./out/{}_L.png'.format(rand_id))


if __name__ == '__main__':
    run()
