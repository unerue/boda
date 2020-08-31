'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math


last_time = time.time()
begin_time = last_time
def progress_bar(epoch, current, total, msg=None):
    global last_time, begin_time

    if epoch == 0 and current == 0:
        header1 = ' Epoch | Step  | Tot      '
        header2 = ' ------+-------+----------'
        for key in msg.keys():
            max_len = 15
            header1 += f'| {key:<{max_len-len(key)}}'
            header2 += '+' + '-'*(max_len-3)

        header = header1 + '\n' + header2
        # header = f' Epoch | Step  | Tot      | Loss  | Acc '
        
        sys.stdout.write(header)
        sys.stdout.write('\n')

    if current == 0:
        begin_time = time.time()  # Reset for new bar.
        
    
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time


    L = []
    # L.append(f' Epoch: {epoch+1:<2}')
    # L.append(f'| Step: {format_time(step_time):<5}')
    # L.append(f'| Tot: {format_time(tot_time):<5}')
    L.append(f' {epoch+1:>4} ')
    L.append(f' | {format_time(step_time):>5}')
    L.append(f' | {format_time(tot_time):>8} ')
    # msg1 = ''
    for item in msg.values():
        max_len = 15 - len(item)
        if max_len < 0:
            max_len = 1
        L.append(f'| {item:>{max_len+2}} ')


    
    # if msg:
    #     L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    # for _ in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
    #     sys.stdout.write(' ')

    # # Go back to the center of the bar.
    # for _ in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
    #     sys.stdout.write('\b')
    # sys.stdout.write(f' {current+1}/{total} ')

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f