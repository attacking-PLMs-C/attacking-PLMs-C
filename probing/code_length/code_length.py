import os 
import json
import pickle as pkl
import tqdm
import re

def get_file_path(root_path, file_list):
    PATH = os.listdir(root_path)
    for path in PATH:
        # print(path)
        co_path = os.path.join(root_path, path)
        if os.path.isfile(co_path):
            file_list.append(co_path)
        elif os.path.isdir(co_path):
            get_file_path(co_path, file_list)
    return file_list

def get_length(code_data):
    code_data = re.sub(r"(\s)+", ' ', code_data).strip()  # [\n ]+
    # with open("example.txt", 'w') as fp:
    #     fp.write(code_data)
    code_data = code_data.split(' ')
    return len(code_data)

if __name__ == "__main__":
    code_path = "../valid_code"
    code_file_list = []
    get_file_path(code_path, code_file_list)
    bar = tqdm.tqdm(code_file_list)
    code_file_length = {}
    cnt = 0
    min = 40
    max = 0
    with open("all_code_length.txt", 'w') as fp:
        for item in bar:
            with open(item, 'r') as fp1:
                # print(item)
                data = fp1.read()
                c_len = get_length(data)
                cname = item.split("/")[-1]
                code_file_length[cname] = c_len
                fp.write(cname + "  " + str(c_len) + '\n')
                if c_len <= 510:
                    cnt += 1
                if min > c_len:
                    min = c_len
                if max < c_len:
                    max = c_len
                bar.update()
        bar.close()
    print(cnt)
    print(min)
    print(max)

    count = 0
    pbar = tqdm.tqdm(code_file_length.items())
    with open("code_length.txt", 'w') as fp:
        for key, val in pbar:
            if val >=3 and val < 8:
                fp.write(key + "  0\n")
            elif val >= 8 and val < 13:
                fp.write(key + "  1\n")
            elif val >= 13 and val < 18:
                fp.write(key + "  2\n")
            elif val >= 18 and val < 23:
                fp.write(key + "  3\n")
            elif val >= 23 and val < 28:
                fp.write(key + "  4\n")
            elif val >= 28 and val < 33:
                fp.write(key + "  5\n")
            elif val >= 33 and val < 38:
                fp.write(key + "  6\n")
            elif val >= 38 and val < 43:
                fp.write(key + "  7\n")
            elif val >= 43 and val < 48:
                fp.write(key + "  8\n")
            elif val >= 48 and val < 53:
                fp.write(key + "  9\n")
            elif val >= 53 and val < 58:
                fp.write(key + "  10\n")
            elif val >= 58 and val < 63:
                fp.write(key + "  11\n")
            elif val >= 63 and val < 68:
                fp.write(key + "  12\n")
            elif val >= 68 and val < 73:
                fp.write(key + "  13\n")
            elif val >= 73 and val < 78:
                fp.write(key + "  14\n")
            elif val >= 78 and val < 83:
                fp.write(key + "  15\n")
            elif val >= 83 and val < 88:
                fp.write(key + "  16\n")
            elif val >= 88 and val < 93:
                fp.write(key + "  17\n")
            elif val >= 93 and val < 98:
                fp.write(key + "  18\n")
            elif val >= 98 and val < 103:
                fp.write(key + "  19\n")
            elif val >= 103 and val < 108:
                fp.write(key + "  20\n")
            elif val >= 108 and val < 113:
                fp.write(key + "  21\n")
            elif val >= 113 and val < 118:
                fp.write(key + "  22\n")
            elif val >= 118 and val < 123:
                fp.write(key + "  23\n")
            elif val >= 123 and val < 128:
                fp.write(key + "  24\n")
            elif val >= 128 and val < 133:
                fp.write(key + "  25\n")
            elif val >= 133 and val < 138:
                fp.write(key + "  26\n")
            elif val >= 138 and val < 143:
                fp.write(key + "  27\n")
            elif val >= 143 and val < 148:
                fp.write(key + "  28\n")
            elif val >= 148 and val < 153:
                fp.write(key + "  29\n")
            elif val >= 153 and val < 158:
                fp.write(key + "  30\n")
            elif val >= 158 and val < 163:
                fp.write(key + "  31\n")
            elif val >= 163 and val < 168:
                fp.write(key + "  32\n")
            elif val >= 168 and val < 173:
                fp.write(key + "  33\n")
            elif val >= 173 and val < 178:
                fp.write(key + "  34\n")
            elif val >= 178 and val < 183:
                fp.write(key + "  35\n")
            elif val >= 183 and val < 188:
                fp.write(key + "  36\n")
            elif val >= 188 and val < 193:
                fp.write(key + "  37\n")
            elif val >= 193 and val < 198:
                fp.write(key + "  38\n")
            elif val >= 198 and val < 203:
                fp.write(key + "  39\n")
            elif val >= 203 and val < 208:
                fp.write(key + "  40\n")
            elif val >= 208 and val < 213:
                fp.write(key + "  41\n")
            elif val >= 213 and val < 218:
                fp.write(key + "  42\n")
            elif val >= 218 and val < 223:
                fp.write(key + "  43\n")
            elif val >= 223 and val < 228:
                fp.write(key + "  44\n")
            elif val >= 228 and val < 233:
                fp.write(key + "  45\n")
            elif val >= 233 and val < 238:
                fp.write(key + "  46\n")
            elif val >= 238 and val < 243:
                fp.write(key + "  47\n")
            elif val >= 243 and val < 248:
                fp.write(key + "  48\n")
            elif val >= 248 and val < 253:
                fp.write(key + "  49\n")
            elif val >= 253 and val < 258:
                fp.write(key + "  50\n")
            elif val >= 258 and val < 263:
                fp.write(key + "  51\n")
            elif val >= 263 and val < 268:
                fp.write(key + "  52\n")
            elif val >= 268 and val < 273:
                fp.write(key + "  53\n")
            elif val >= 273 and val < 278:
                fp.write(key + "  54\n")
            elif val >= 278 and val < 283:
                fp.write(key + "  55\n")
            elif val >= 283 and val < 288:
                fp.write(key + "  56\n")
            elif val >= 288 and val < 293:
                fp.write(key + "  57\n")
            elif val >= 293 and val < 298:
                fp.write(key + "  58\n")
            elif val >= 298 and val < 303:
                fp.write(key + "  59\n")
            elif val >= 303 and val < 308:
                fp.write(key + "  60\n")
            elif val >= 308 and val < 313:
                fp.write(key + "  61\n")
            elif val >= 313 and val < 318:
                fp.write(key + "  62\n")
            elif val >= 318 and val < 323:
                fp.write(key + "  63\n")
            elif val >= 323 and val < 328:
                fp.write(key + "  64\n")
            elif val >= 328 and val < 333:
                fp.write(key + "  65\n")
            elif val >= 333 and val < 338:
                fp.write(key + "  66\n")
            elif val >= 338 and val < 343:
                fp.write(key + "  67\n")
            elif val >= 343 and val < 348:
                fp.write(key + "  68\n")
            elif val >= 348 and val < 353:
                fp.write(key + "  69\n")
            elif val >= 353 and val < 358:
                fp.write(key + "  70\n")
            elif val >= 358 and val < 363:
                fp.write(key + "  71\n")
            elif val >= 363 and val < 368:
                fp.write(key + "  72\n")
            elif val >= 368 and val < 373:
                fp.write(key + "  73\n")
            elif val >= 373 and val < 378:
                fp.write(key + "  74\n")
            elif val >= 378 and val < 383:
                fp.write(key + "  75\n")
            elif val >= 383 and val < 388:
                fp.write(key + "  76\n")
            elif val >= 388 and val < 393:
                fp.write(key + "  77\n")
            elif val >= 393 and val < 398:
                fp.write(key + "  78\n")
            elif val >= 398 and val <= 403:
                fp.write(key + "  79\n")
            if val > 403:
                count += 1
            pbar.update()
        pbar.close()
        print('**', count, '**')