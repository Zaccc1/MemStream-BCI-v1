"""
Test script: find channel brain region labels for an IBL session.
Run this standalone to see what data is available before modifying the pipeline.

Usage: python test_channel_regions.py
"""

import numpy as np

SESSION_EID = "3f6e25ae-c007-4dc3-aa77-450fd5705046"
PROBE_LABEL = "probe00"
IBL_BASE_URL = "https://openalyx.internationalbrainlab.org"

def connect():
    from one.api import ONE
    one = ONE(base_url=IBL_BASE_URL, password="international", silent=True)
    print(f"Connected to {IBL_BASE_URL}")
    return one

def test_channels_object(one, eid, probe):
    """Test 1: load channels object and inspect all attributes."""
    print(f"\n{'='*60}")
    print(f"  TEST 1: channels object attributes")
    print(f"{'='*60}")
    collections = [
        f'alf/{probe}',
        f'alf/{probe}/pykilosort',
        f'alf/{probe}/iblsorter',
    ]
    for coll in collections:
        try:
            ch = one.load_object(eid, 'channels', collection=coll)
            attrs = [a for a in dir(ch) if not a.startswith('_')]
            print(f"\n  Collection '{coll}':")
            print(f"    Attributes: {attrs}")
            for attr in attrs:
                val = getattr(ch, attr, None)
                if val is not None:
                    if hasattr(val, 'shape'):
                        print(f"    {attr}: shape={val.shape}, dtype={val.dtype}, sample={val[:3]}")
                    elif hasattr(val, '__len__'):
                        print(f"    {attr}: len={len(val)}, sample={list(val[:3])}")
                    else:
                        print(f"    {attr}: {val}")
        except Exception as e:
            print(f"\n  Collection '{coll}': FAILED - {e}")

def test_direct_dataset(one, eid, probe):
    """Test 2: try to load specific dataset files by name."""
    print(f"\n{'='*60}")
    print(f"  TEST 2: direct dataset loading")
    print(f"{'='*60}")
    dataset_names = [
        'channels.brainLocationAcronyms_ccf_2017.npy',
        'channels.brainLocationIds_ccf_2017.npy',
        'channels.acronym.npy',
        'channels.atlas_id.npy',
        'channels.brain_region.npy',
    ]
    collections = [
        f'alf/{probe}',
        f'alf/{probe}/pykilosort',
        f'alf/{probe}/iblsorter',
    ]
    for coll in collections:
        print(f"\n  Collection '{coll}':")
        for ds in dataset_names:
            try:
                data = one.load_dataset(eid, ds, collection=coll)
                if data is not None:
                    if hasattr(data, 'shape'):
                        print(f"    {ds}: shape={data.shape}, dtype={data.dtype}")
                        print(f"      sample: {data[:5]}")
                    else:
                        print(f"    {ds}: {type(data)}, len={len(data)}, sample={list(data[:5])}")
                else:
                    print(f"    {ds}: returned None")
            except Exception as e:
                err_msg = str(e)[:80]
                print(f"    {ds}: FAILED - {err_msg}")

def test_list_datasets(one, eid, probe):
    """Test 3: list all available datasets for this probe."""
    print(f"\n{'='*60}")
    print(f"  TEST 3: all available datasets for {probe}")
    print(f"{'='*60}")
    try:
        datasets = one.list_datasets(eid)
        probe_datasets = [d for d in datasets if probe in str(d)]
        channel_datasets = [d for d in probe_datasets if 'channel' in str(d).lower()]
        print(f"\n  All probe datasets ({len(probe_datasets)}):")
        for d in sorted(probe_datasets):
            print(f"    {d}")
        print(f"\n  Channel-related ({len(channel_datasets)}):")
        for d in sorted(channel_datasets):
            print(f"    {d}")
    except Exception as e:
        print(f"  FAILED: {e}")

def test_spike_sorting_loader(one, eid, probe):
    """Test 4: try brainbox SpikeSortingLoader."""
    print(f"\n{'='*60}")
    print(f"  TEST 4: SpikeSortingLoader")
    print(f"{'='*60}")
    try:
        from brainbox.io.one import SpikeSortingLoader
        ssl = SpikeSortingLoader(one=one, eid=eid, pname=probe)
        print(f"  SpikeSortingLoader created")
        print(f"  Available attributes: {[a for a in dir(ssl) if not a.startswith('_')]}")
        
        # Try loading channels
        try:
            spikes, clusters, channels = ssl.load_spike_sorting()
            print(f"\n  Spikes keys: {list(spikes.keys())}")
            print(f"  Clusters keys: {list(clusters.keys())}")
            print(f"  Channels keys: {list(channels.keys())}")
            
            for key in channels:
                val = channels[key]
                if hasattr(val, 'shape'):
                    print(f"    channels.{key}: shape={val.shape}, dtype={val.dtype}, sample={val[:3]}")
                elif hasattr(val, '__len__') and len(val) > 0:
                    print(f"    channels.{key}: len={len(val)}, sample={list(val[:3])}")
        except Exception as e:
            print(f"  load_spike_sorting failed: {e}")
            
        # Try get_channels
        try:
            channels2 = ssl.load_channels()
            print(f"\n  load_channels keys: {list(channels2.keys()) if isinstance(channels2, dict) else dir(channels2)}")
        except Exception as e:
            print(f"  load_channels failed: {e}")
            
    except ImportError:
        print("  brainbox not available (pip install ibllib)")
    except Exception as e:
        print(f"  FAILED: {e}")

def test_atlas_mapping(one, eid, probe):
    """Test 5: try atlas ID -> acronym mapping via iblatlas."""
    print(f"\n{'='*60}")
    print(f"  TEST 5: atlas ID mapping")
    print(f"{'='*60}")
    
    # First try to get atlas IDs
    atlas_ids = None
    for coll in [f'alf/{probe}/pykilosort', f'alf/{probe}', f'alf/{probe}/iblsorter']:
        for ds in ['channels.brainLocationIds_ccf_2017.npy', 'channels.atlas_id.npy']:
            try:
                atlas_ids = one.load_dataset(eid, ds, collection=coll)
                if atlas_ids is not None:
                    print(f"  Loaded atlas IDs from {coll}/{ds}: shape={atlas_ids.shape}")
                    break
            except:
                pass
        if atlas_ids is not None:
            break
    
    if atlas_ids is None:
        print("  No atlas IDs found")
        return
    
    # Try to map IDs to acronyms
    try:
        from iblatlas.regions import BrainRegions
        br = BrainRegions()
        acronyms = br.id2acronym(atlas_ids)
        print(f"  Mapped {len(acronyms)} IDs to acronyms")
        print(f"  Sample: {list(acronyms[:10])}")
        from collections import Counter
        counts = Counter(acronyms)
        print(f"  Top 10 regions: {counts.most_common(10)}")
    except ImportError:
        print("  iblatlas not available (pip install iblatlas)")
    except Exception as e:
        print(f"  Mapping failed: {e}")


if __name__ == '__main__':
    one = connect()
    eid = SESSION_EID
    probe = PROBE_LABEL
    
    test_channels_object(one, eid, probe)
    test_direct_dataset(one, eid, probe)
    test_list_datasets(one, eid, probe)
    test_spike_sorting_loader(one, eid, probe)
    test_atlas_mapping(one, eid, probe)
    
    print(f"\n{'='*60}")
    print(f"  DONE — check which test found the brain regions")
    print(f"{'='*60}")
