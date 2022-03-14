import torch
import numpy
import os
import fnmatch
import csv

class LogWriter(object):

    def __init__(self, path, args):
        if '' in args:
            del args['']
        self.path = path
        self.args = args
        with open(self.path, 'w') as f:
            f.write("Training Log\n")
            f.write("Specifications\n")
            for argname in self.args:
                f.write("{} : {}\n".format(argname, self.args[argname]))
            f.write("Checkpoints:\n")

    def checkpoint(self, to_write):
        with open(self.path, 'a') as f:
            f.write(to_write+'\n')

    def initBest(self):
        self.current_best = {
            'loglik': numpy.finfo(float).min,
            'distance': numpy.finfo(float).max
        }
        self.episode_best = 'NeverUpdated'

    def updateBest(self, key, value, episode):
        updated = False
        if key == 'loglik':
            if value > self.current_best[key]:
                updated = True
                self.current_best[key] = value
                self.episode_best = episode
        elif key == 'distance':
            if value < self.current_best[key]:
                updated = True
                self.current_best[key] = value
                self.episode_best = episode
        else:
            raise Exception("unknown key {}".format(key))
        return updated


class LogReader(object):

    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            self.doc = f.read()

    def isfloat(self, str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    def casttype(self, str):
        res = None
        if str.isdigit():
            res = int(str)
        elif self.isfloat(str):
            res = float(str)
        elif str == 'True' or str == 'False':
            res = True if str == 'True' else False
        else:
            res = str
        return res

    def finished(self): 
        return 'training finished' in self.doc

    def getArgs(self):
        block_args = self.doc.split('Specifications\n')[-1]
        block_args = block_args.split('Checkpoints:\n')[0]
        lines_args = block_args.split('\n')
        res = {}
        for line in lines_args:
            items = line.split(' : ')
            res[items[0]] = self.casttype(items[-1])
        return res

    def getBest(self):
        block_score = self.doc.split('Checkpoints:\n')[-1]
        lines_score = block_score.split('\n')
        best_score = ''
        best_episode = ''
        for line in lines_score:
            if 'current best loglik is' in line:
                best_score = line.split('current best loglik is ')[-1]
                best_score = best_score.split(' (updated at')[0]
                best_episode = line.split('(updated at episode-')[-1]
                best_episode = best_episode.split(')')[0]
                best_score = self.casttype(best_score)
                best_episode = self.casttype(best_episode)
        return best_score, best_episode

    def getAll(self):
        res = self.getArgs()
        best_score, best_episode = self.getBest()
        res['_best_score'] = best_score
        res['_best_episode'] = best_episode
        return res


class LogBatchReader(object):

    def __init__(self, path):
        self.path = path
        """
        given domain path, find all the log folders and get their results
        """
        self.all_readers = []
        for dirpath, dirname, files in os.walk(self.path):
            for file_name in fnmatch.filter(files, 'log.txt'):
                full_path = os.path.join(dirpath, file_name)
                if 'Logs/' in full_path: 
                    # don't read logs in backup folders
                    self.all_readers.append( LogReader(full_path) )

    def writeCSV(self):
        path_save = os.path.join(self.path, 'logs.csv')
        print(f"writing CSV to {path_save}")
        names_field = self.makeHeader()
        c = 0
        with open(path_save, 'w') as file_csv:
            writer_csv = csv.DictWriter( file_csv, fieldnames = names_field )
            writer_csv.writeheader()
            for log_reader in self.all_readers:
                args = log_reader.getAll()
                self.processInfo(args)
                writer_csv.writerow(args)
                if log_reader.finished(): 
                    c += 1 
        print(f"CSV finished")
        print(f"{c} out of {len(self.all_readers)} organized logs finished training")

    def makeHeader(self):
        names_field = set()
        for log_reader in self.all_readers:
            args = log_reader.getAll()
            self.processInfo(args)
            for k in args:
                names_field.add(k)
        return sorted(list(names_field))

    def processInfo(self, args):
        del args['Database']
        del args['PathStorage']
        del args['LearnRate']
        del args['UseGPU']
        del args['Seed']
        del args['Version']
        del args['ID']
        del args['TIME']
        del args['PathDomain']
        del args['PathLog']
        args['FolderName'] = args['PathSave'].split('/saved_model')[0].split('Logs/')[-1]
        del args['PathSave']
        del args['NumProcess']
        del args['NumThread']
        if '' in args: 
            del args['']
