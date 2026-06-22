import numpy as np
import os
import datetime
from framework import data_types

def read(filename:str, setup:str):
    '''
    :param filename: path where the .lvm is stored
    :param setup: define which type of test the data was taken from
    :param timezone: timezone to convert unix timestamp to human timestamp
    '''

    #check if the filename exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'[file_lvm] Filename {filename} does not exist!')

    setup = setup.lower()  # convert to lowercase
    valid_setup = ["freezer", "lulacell"]
    if setup not in valid_setup:
        raise ValueError(f'[readfileLVM] "setup" = {setup} not implemented! Try: {valid_setup}')

    #process the raw data output from the PHOBOS acquisition system into a custom data structure
    with open(filename, 'r') as raw_file:
        end_header = 0 #monitor the headers
        date_line = ""
        time_line = ""
        raw_data = [] #list to append all the remaining raw lines after the headers
        header_cleared = False #flag to monitor when the header is fully read
        timestamp_defined = False #flag to monitor when the initial timestamp is defined
        for line in raw_file:
            #skip two sets of headers
            if not header_cleared:
                if type(line) != str:
                    raise ValueError(f'[file_lvm] No headers found for the LVM file!')

                #store the date and time entries for timestamp syncing (last entry)
                if line.startswith('Date'):
                    date_line = line
                elif line.startswith('Time'):
                    time_line = line

                if line.startswith('Channels'):
                    channels = line

                #handle header identification
                if "***End_of_Header***" in line:
                    end_header += 1 #update the header counter

                if end_header == 2:
                    header_cleared = True #commute the flag
                    date_line = date_line.strip('\n') #strip of any line breakers
                    time_line = time_line.strip('\n') #strip of any line breakers
                    channels = channels.strip('\n')
            else:
                if not timestamp_defined:
                    if date_line == '':
                        print(f'[file_lvm] Failed to reference Date/Time, returning relative time sync!')
                        lvm_start_stamp = None
                    else:
                        date_line = date_line.split('\t') #split on tabs
                        time_line = time_line.split('\t') #split on tabs
                        channels = int(channels.split('\t')[1])
                        #handle unexpected formats
                        if len(date_line)<2 or len(time_line)<2:
                            print(f'[file_lvm] Unexpected format in the header, returning relative time sync!')
                            lvm_start_stamp = 0 #relative timestamp
                        else:
                            date_init = date_line[1] #extract the date from the list of dates
                            time_init = time_line[1] #extract the time from the list of timestamps
                            if setup == "freezer":
                                time_init_split = time_init.split(',')
                            else:
                                time_init_split = time_init.split('.') #separate integer to decimal part (',' as decimal notation)
                            timestamp_init = date_init + " " + time_init_split[0] + "," + time_init_split[1][:3]
                            timestamp_init = datetime.datetime.strptime(timestamp_init, "%Y/%m/%d %H:%M:%S,%f")
                            second_delta = datetime.timedelta(seconds=float(timestamp_init.second + timestamp_init.microsecond*1e-6)) #floating point seconds
                            timestamp_init += second_delta #add the floating point seconds to the object
                            timestamp_defined = True #commute the flag
                else:
                    if type(line) != str:
                        break
                    raw_data.append(line) #add a line to the list

        raw_file.close() #close the reader
        raw_data = np.array(raw_data) #convert to numpy array
        if setup == "freezer":
            data = data_types.TemperatureData(raw_data, timestamp_init, channels) #organize the raw data
        else:
            data = data_types.TempPressData(raw_data, timestamp_init)
        return data




