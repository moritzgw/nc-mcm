# License: BSD 3-clause
# Author: Boris Reuderink

import logging, re, os.path, StringIO, itertools
from ConfigParser import SafeConfigParser
import numpy as np

# TODO:
# - add encoding of commas (\1)
# - verify units for resolution in UTF8

log = logging.getLogger('__main__')

def read_header(fname):
  with open(fname) as f:
    # setup config reader
    assert f.readline().strip() == \
      'Brain Vision Data Exchange Header File Version 1.0'

    lines = itertools.takewhile(lambda x: '[Comment]' not in x, f.readlines())
    cfg = SafeConfigParser()
    cfg.readfp(StringIO.StringIO(''.join(lines)))

    # get sampling info
    sample_rate = 1e6/cfg.getfloat('Common Infos', 'SamplingInterval')
    nchann = cfg.getint('Common Infos', 'NumberOfChannels')

    log.info('Found sample rate of %.2f Hz, %d channels.' % 
      (sample_rate, nchann))

    # check binary format
    assert cfg.get('Common Infos', 'DataOrientation') == 'MULTIPLEXED'
    assert cfg.get('Common Infos', 'DataFormat') == 'BINARY'
    assert cfg.get('Binary Infos', 'BinaryFormat') == 'INT_16'

    # load channel labels
    chan_lab = ['UNKNOWN'] * nchann
    chan_resolution = np.ones(nchann) * np.nan
    for (chan, props) in cfg.items('Channel Infos'):
      n = int(re.match(r'ch(\d+)', chan).group(1))
      name, refname, resolution, unit = props.split(',')[:4]

      chan_lab[n-1] = name
      chan_resolution[n-1] = resolution

    # locate EEG and marker files
    eeg_fname = cfg.get('Common Infos', 'DataFile')
    marker_fname = cfg.get('Common Infos', 'MarkerFile')

    return dict(sample_rate=sample_rate, chan_lab=chan_lab, 
      chan_resolution=chan_resolution, eeg_fname=eeg_fname, 
      marker_fname=marker_fname)


def read_eeg(fname, chan_resolution):
  chan_resolution = np.atleast_2d(chan_resolution)
  nchan = chan_resolution.size

  with open(fname, 'rb') as f:
    raw = f.read()
    size = len(raw)/2
    ints = np.ndarray((nchan, size/nchan), dtype='<i2', order='F', buffer=raw)
    return ints * chan_resolution.T


def read_markers(fname):
  with open(fname) as f:
    # setup config reader
    assert f.readline().strip() == \
      'Brain Vision Data Exchange Marker File, Version 1.0'

    cfg = SafeConfigParser()
    cfg.readfp(f)
    events = []
    for (marker, info) in cfg.items('Marker Infos'):
      mtype, mdesc, offset, duration, channels = info.split(',')[:5]
      if mtype == 'Stimulus':
        stim_code = int(re.match(r'S\s+(\d+)', mdesc).group(1))
        offset, duration = int(offset), int(duration)
        events.append((stim_code, offset, offset+duration))
      else:
        log.warning(
          'Skipping unknown marker %s of type "%s".' % (marker, mtype))
    events = np.asarray(events, int).T
    events[1:] -= 1  # use zero-based indexing
    return events


def read_brainvis_triplet(header_fname, marker_fname=None, eeg_fname=None):
  '''
  Read BrainVision Recorder header file, locate and read the marker and EEG
  file. Returns a header dictionary, a matrix with events and the raw EEG.
  '''
  containing_dir = os.path.split(header_fname)[0]
  header = read_header(header_fname)

  if not marker_fname:
    # locate marker file
    marker_fname = os.path.join(containing_dir, header['marker_fname'])
  E = read_markers(marker_fname)

  if not eeg_fname:
    # locate EEG file
    eeg_fname = os.path.join(containing_dir, header['eeg_fname'])
  X = read_eeg(eeg_fname, header['chan_resolution'])

  return header, E, X