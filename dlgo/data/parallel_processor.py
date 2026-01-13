from __future__ import print_function
from __future__ import absolute_import
import os
import glob
import os.path
import tarfile
import gzip
import shutil
import numpy as np
import multiprocessing
from os import sys


from dlgo.gosgf.sgf import Sgf_game
from dlgo.goboard import Board, GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.data.index_processor import KGSIndex
from dlgo.data.sampling import Sampler
from dlgo.data.generator import DataGenerator
from dlgo.encoders.base import get_encoder_by_name


def worker(jobinfo):
    try:
        clazz, encoder, zip_file, data_file_name, game_list = jobinfo
        clazz(encoder=encoder).process_zip(zip_file, data_file_name, game_list)
    except (KeyboardInterrupt, SystemExit):
        raise Exception('>>> Exiting child process.')


class GoDataProcessor:
    def __init__(self, encoder='simple', data_directory='./dlgo/data/sgfdata'):
        self.encoder_string = encoder
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = data_directory

    def load_go_data(self, data_type='train', num_samples=1000, use_generator=False):
        index = KGSIndex(data_directory=self.data_dir)
        # index.download_files()

        sampler = Sampler(data_dir=self.data_dir)
        data = sampler.draw_data(data_type, num_samples)

        self.map_to_workers(data_type, data)  
        if use_generator:
            generator = DataGenerator(self.data_dir, data)
            return generator  
        else:
            if num_samples > 1000:
                print(f"Warning: Consolidating {num_samples} samples will use a lot of memory!")
                print("Consider using use_generator=True instead.")
        
            features_and_labels = self.consolidate_games(data_type, data)
            return features_and_labels


    def unzip_data(self, zip_file_name):
        this_gz = gzip.open(self.data_dir + '/' + zip_file_name)

        tar_file = zip_file_name[0:-3]
        this_tar = open(self.data_dir + '/' + tar_file, 'wb')

        shutil.copyfileobj(this_gz, this_tar)
        this_tar.close()
        return tar_file

    def process_zip(self, zip_file_name, data_file_name, game_list):
        """Process zip file with minimal memory usage by streaming to disk."""
        tar_file = self.unzip_data(zip_file_name)
        zip_file = tarfile.open(self.data_dir + '/' + tar_file)
        name_list = zip_file.getnames()
        
        feature_file_base = self.data_dir + '/' + data_file_name + '_features_%d'
        label_file_base = self.data_dir + '/' + data_file_name + '_labels_%d'
        
        # Accumulate features in smaller chunks
        chunk = 0
        chunksize = 1024
        features_buffer = []
        labels_buffer = []
        
        for index in game_list:
            name = name_list[index + 1]
            if not name.endswith('.sgf'):
                continue
                
            try:
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content)
            except Exception as e:
                print(f"Error reading {name}: {e}")
                continue

            game_state, first_move_done = self.get_handicap(sgf)

            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:
                        row, col = move_tuple
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)
                    else:
                        move = Move.pass_turn()
                        
                    if first_move_done and point is not None:
                        features_buffer.append(self.encoder.encode(game_state))
                        labels_buffer.append(self.encoder.encode_point(point))
                        
                        # Save chunk when buffer is full
                        if len(features_buffer) >= chunksize:
                            feature_file = feature_file_base % chunk
                            label_file = label_file_base % chunk
                            
                            np.save(feature_file, np.array(features_buffer, dtype=np.float32))
                            np.save(label_file, np.array(labels_buffer, dtype=np.int64))
                            
                            print(f"Saved chunk {chunk} with {len(features_buffer)} positions")
                            
                            features_buffer = []
                            labels_buffer = []
                            chunk += 1
                        
                    game_state = game_state.apply_move(move)
                    first_move_done = True
        
        # Save remaining buffer
        if features_buffer:
            feature_file = feature_file_base % chunk
            label_file = label_file_base % chunk
            np.save(feature_file, np.array(features_buffer, dtype=np.float32))
            np.save(label_file, np.array(labels_buffer, dtype=np.int64))
            print(f"Saved final chunk {chunk} with {len(features_buffer)} positions")
        
        zip_file.close()

    def consolidate_games(self, name, samples):
        files_needed = set(file_name for file_name, index in samples)
        file_names = []
        for zip_file_name in files_needed:
            file_name = zip_file_name.replace('.tar.gz', '') + name
            file_names.append(file_name)

        feature_list = []
        label_list = []
        for file_name in file_names:
            file_prefix = file_name.replace('.tar.gz', '')
            base = self.data_dir + '/' + file_prefix + '_features_*.npy'
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                y = y.astype(np.int64)
                feature_list.append(x)
                label_list.append(y)

        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        feature_file = self.data_dir + '/' + name
        label_file = self.data_dir + '/' + name

        np.save(feature_file, features)
        np.save(label_file, labels)

        return features, labels

    @staticmethod
    def get_handicap(sgf):  # Get handicap stones
        go_board = Board(19, 19)
        first_move_done = False
        move = None
        game_state = GameState.new_game(19)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))  # black gets handicap
            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move)
        return game_state, first_move_done

    def map_to_workers(self, data_type, samples):
        zip_names = set()
        indices_by_zip_name = {}
        for filename, index in samples:
            zip_names.add(filename)
            if filename not in indices_by_zip_name:
                indices_by_zip_name[filename] = []
            indices_by_zip_name[filename].append(index)

        zips_to_process = []
        for zip_name in zip_names:
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            
            # Check if chunk files exist instead of consolidated file
            chunk_pattern = self.data_dir + '/' + data_file_name + '_features_0.npy'
            
            if not os.path.isfile(chunk_pattern):
                print(f"Need to process: {zip_name} for {data_type}")
                zips_to_process.append((self.__class__, self.encoder_string, zip_name,
                                        data_file_name, indices_by_zip_name[zip_name]))
            else:
                print(f"Already processed: {zip_name} for {data_type}")

        if not zips_to_process:
            print(f"All {data_type} data already processed")
            return

        cores = multiprocessing.cpu_count()
        # Limit cores to reduce memory usage
        cores = min(cores, 4)  # Use max 4 cores
        
        print(f"Processing {len(zips_to_process)} files with {cores} workers...")
        
        pool = multiprocessing.Pool(processes=cores)
        p = pool.map_async(worker, zips_to_process)
        try:
            _ = p.get()
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            sys.exit(-1)

    def num_total_examples(self, zip_file, game_list, name_list):
        total_examples = 0
        for index in game_list:
            name = name_list[index + 1]
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content)
                game_state, first_move_done = self.get_handicap(sgf)

                num_moves = 0
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_examples = total_examples + num_moves
            else:
                raise ValueError(name + ' is not a valid sgf')
        return total_examples