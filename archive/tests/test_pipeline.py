"""No training/backpropagation. Pure preprocessing and forward/checkpoint checks."""
from pathlib import Path
import importlib.util,json,sys,tempfile,unittest
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ev3dar.common import load_config,write_csv,data_path
from ev3dar.data import make_split,source_audit
from ev3dar.features import (circular_segments,angle_track,extract_record,fit_normalization,
                             cache_path,feature_key)
from ev3dar.metrics import (circular_peaks,gaussian_target,angular_error,
                            classification_metrics,spatial_map)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'.test_tmp').mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.test_tmp')
        self.path=Path(self.temp.name);self.cfg=load_config()
    def tearDown(self):self.temp.cleanup()
    def test_portable_path_and_escape(self):
        self.assertEqual(data_path(self.path,'측정_데이터\\a.wav'),self.path/'측정_데이터/a.wav')
        for p in ['../a.wav','C:/a.wav','/tmp/a.wav']:
            with self.assertRaises(ValueError):data_path(self.path,p)
    def test_zero_degree_peak_and_missing_source_penalty(self):
        target=gaussian_target([0,180,270])
        self.assertEqual(set(circular_peaks(target,3)),{0,180,270})
        self.assertEqual(angular_error([], [0,180,270]),180.)
        self.assertEqual(angular_error([0,180],[0,180,270]),60.)
    def test_wrap_separate_views_and_nonuniform_angle_time(self):
        cfg=self.cfg;fs=cfg['sample_rate'];t=np.arange(fs*4)/fs
        sf.write(self.path/'x.wav',.1*np.sin(2*np.pi*1700*t),fs,subtype='PCM_16')
        # First half time covers only90deg; naive four equal time slices are wrong.
        theta=np.interp(np.linspace(0,4,401),[0,2,4],[0,90,360])
        write_csv(self.path/'a.csv',[dict(elapsed_sec=float(i/100),angle_deg=float(a)) for i,a in enumerate(theta)])
        r=dict(uid='test_0001',wav='x.wav',angles='a.csv',duration=4.,rpm_intervals=[[5000,7000]]*3)
        f=extract_record(self.path,r,cfg)
        self.assertEqual(circular_segments(315,405),[(315,360.),(0.,45.)])
        self.assertEqual(f['ssl'].shape,(128,360))
        self.assertEqual(f['id_views'].shape,(3,2,64,100))
        # REAR has two genuine views. FRONT and RIGHT have only one.
        np.testing.assert_array_equal((f['view_durations']>0).sum(1),[1,2,1])
        self.assertAlmostEqual(float(f['view_durations'][1,0]),1/3,places=2)
        self.assertAlmostEqual(float(f['view_durations'][1,1]),1.,places=2)
        self.assertTrue(np.isfinite(f['ssl']).all())
    def test_discontinuity_rejected(self):
        write_csv(self.path/'bad.csv',[dict(elapsed_sec=i/100,angle_deg=(i if i!=15 else 200)) for i in range(40)])
        with self.assertRaises(ValueError):angle_track(self.path/'bad.csv',.4)
    def test_split_before_windows_and_source_overlap_is_reported(self):
        records=[]
        for f in range(4):
            for band in ['A','B']:
                for i in range(150):records.append(dict(uid=f'{f}_{band}_{i:03}',fault_count=f,band=band,source_files=['shared']))
        for seed in [42,43,44]:
            split=make_split(records,seed)
            self.assertEqual([len(split[k]) for k in ['train','validation','test']],[960,120,120])
            self.assertEqual(split['source_file_overlap']['train__test'],1)
            self.assertFalse(split['unseen_source_generalization_claim_allowed'])
        self.assertEqual(source_audit(records)['connected_component_sizes'],[1200])
        self.assertEqual(make_split(records,42),make_split(records,42))
        self.assertNotEqual(make_split(records,42)['test'],make_split(records,43)['test'])
    def test_metrics_use_real_top1(self):
        # Truth appears among top3 but NOT top1: it must remain wrong.
        p=np.array([[.1,.6,.2,.04,.03,.03]])
        m=classification_metrics([2],p,self.cfg['classes'])
        self.assertEqual(m['type_accuracy'],0.)
        self.assertEqual(m['type_confusion_matrix'][2][1],1)
    def test_normalization_uses_only_supplied_training_records(self):
        (self.path/'cache').mkdir()
        for uid,value in [('train',1.),('validation',1000.)]:
            np.savez(cache_path(self.path,uid),ssl=np.full((128,360),value),
                     id_views=np.full((3,2,64,100),value),id_masks=np.ones((3,2,100)))
        norm=fit_normalization([dict(uid='train')],self.path)
        self.assertTrue(np.allclose(norm['ssl']['mean'],1.))
        self.assertTrue(np.allclose(norm['id']['mean'],1.))
        self.assertEqual(norm['fitted_recording_ids'],['train'])
    def test_spatial_map_omits_excluded_left_region(self):
        p=np.eye(6)[[1,2,4]];out=spatial_map(np.ones(360),p,self.cfg)
        np.testing.assert_array_equal(out[90],np.zeros(6))
        np.testing.assert_array_equal(out[0],p[1])
        np.testing.assert_array_equal(out[180],p[0])
        np.testing.assert_array_equal(out[270],p[2])
    def test_feature_cache_independent_of_training_epochs(self):
        r=dict(uid='a',input_fingerprint={})
        cfg=dict(self.cfg);cfg['epochs']=1
        self.assertEqual(feature_key(r,cfg),feature_key(r,self.cfg))


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable; run forward tests on Spark.')
class ForwardOnlyTests(unittest.TestCase):
    def test_models_and_checkpoint_roundtrip_without_training(self):
        import torch
        from ev3dar.models import LocalizationCNN,SoundTypeCNN
        torch.set_num_threads(2)
        (ROOT/'.test_tmp').mkdir(exist_ok=True)
        with torch.no_grad():
            ssl=LocalizationCNN(128).eval();counts,angles=ssl(torch.randn(2,128,360))
            self.assertEqual(tuple(counts.shape),(2,3));self.assertEqual(tuple(angles.shape),(2,360))
            views=torch.randn(2,3,2,64,100);masks=torch.ones(2,3,2,100);masks[:,:,1]=0
            durations=torch.ones(2,3,2);durations[:,:,1]=0;rpm=torch.ones(2,3,2)
            for use in [False,True]:
                net=SoundTypeCNN(use_rpm=use).eval();logits,latent=net(views,masks,durations,rpm)
                self.assertEqual(tuple(logits.shape),(2,3,6));self.assertEqual(tuple(latent.shape),(2,3,64))
                self.assertTrue(torch.isfinite(logits).all())
                with tempfile.TemporaryDirectory(dir=ROOT/'.test_tmp') as folder:
                    p=Path(folder)/'weights.pt';torch.save(net.state_dict(),p)
                    clone=SoundTypeCNN(use_rpm=use).eval();clone.load_state_dict(torch.load(p,weights_only=True))
                    other,_=clone(views,masks,durations,rpm);torch.testing.assert_close(other,logits)

    def test_untrained_evaluation_output_contract(self):
        # Synthetic features + random weights only; NOT a performance experiment.
        import torch
        from ev3dar.models import LocalizationCNN,SoundTypeCNN
        from ev3dar.engine import evaluate
        from ev3dar.common import write_json,config_hash,digest
        torch.set_num_threads(2);cfg=load_config()
        (ROOT/'.test_tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'.test_tmp') as folder:
            out=Path(folder);(out/'cache').mkdir();run=out/'seed42';run.mkdir()
            records=[dict(uid=f'fixture_{i}',classes=classes,source_count=3,input_fingerprint={},wav='synthetic.wav',band='fixture')
                     for i,classes in enumerate([['Healthy','Imbalance','Wind'],['Whine','Healthy','PWM']])]
            training_record=dict(uid='training_fixture',classes=['Healthy','Healthy','Wind'])
            (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in [*records,training_record]),encoding='utf-8')
            rng=np.random.default_rng(7)
            for r in records:
                np.savez(cache_path(out,r['uid']),ssl=rng.normal(size=(128,360)).astype('float32'),
                    id_views=rng.normal(size=(3,2,64,100)).astype('float32'),id_masks=np.ones((3,2,100),np.float32),
                    view_durations=np.ones((3,2),np.float32),rpm=np.ones((3,2),np.float32),feature_key=feature_key(r,cfg))
            norm=dict(ssl=dict(mean=[0.]*128,std=[1.]*128),id=dict(mean=[0.]*64,std=[1.]*64))
            split=dict(train=['training_fixture'],test=[r['uid'] for r in records],source_file_overlap={},mode='synthetic_test_only')
            for task,model in [('id',SoundTypeCNN()),('ssl',LocalizationCNN(128))]:
                torch.save(dict(state_dict=model.state_dict(),task=task,config=cfg,config_hash=config_hash(cfg),
                    normalization=norm,split=split,manifest_sha256=digest(out/'manifest.jsonl')),run/(task+'_best.pt'))
            metrics=evaluate(out,cfg,42,device='cpu')
            self.assertEqual(metrics['n_recordings'],2)
            self.assertIn('training_position_prior_baseline',metrics)
            self.assertTrue((run/'test_predictions.csv').is_file())
            self.assertEqual(np.load(run/'spatial_maps/fixture_0.npz')['joint_map'].shape,(360,6))


if __name__=='__main__':unittest.main(verbosity=2)
