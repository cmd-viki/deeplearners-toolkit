import unittest
import tempfile
import os
import numpy as np
from numcompute_stream import DecisionTreeClassifier, StandardScaler, Imputer, OneHotEncoder, Pipeline, StreamingMetrics, StreamingStats, StreamTrainer
from numcompute_stream.io import read_csv,iter_csv_chunks

X=np.array([[0,0],[1,0],[2,1],[3,1],[4,0],[5,1]],float)
y=np.array([0,0,0,1,1,1])
class TestTree(unittest.TestCase):
    def test_fit_predict(self): self.assertTrue(np.array_equal(DecisionTreeClassifier().fit(X,y).predict(X),y))
    def test_score(self): self.assertEqual(DecisionTreeClassifier().fit(X,y).score(X,y),1.)
    def test_buffer_wait(self):
        t=DecisionTreeClassifier(buffer_size=6).partial_fit(X[:3],y[:3]); self.assertIsNone(t.root_)
    def test_buffer_flush(self):
        t=DecisionTreeClassifier(buffer_size=6).partial_fit(X,y); self.assertEqual(t.buffered_samples_,0)
    def test_retrain_count(self):
        t=DecisionTreeClassifier(buffer_size=3)
        t.partial_fit(X[:3],y[:3]);t.partial_fit(X[3:],y[3:]);self.assertEqual(t.retrain_count_,2)
    def test_old_tree_between_retrains(self):
        t=DecisionTreeClassifier(buffer_size=3).partial_fit(X[:3],y[:3]); a=t.predict(X)
        t.partial_fit(X[3:5],y[3:5]);np.testing.assert_array_equal(t.predict(X),a)
    def test_oversize_chunk(self): self.assertEqual(DecisionTreeClassifier(buffer_size=2).partial_fit(X,y).retrain_count_,1)
    def test_nan(self):
        xx=X.copy();xx[0,0]=np.nan
        self.assertEqual(len(DecisionTreeClassifier().fit(xx,y).predict(xx)),6)
    def test_entropy(self): self.assertEqual(len(DecisionTreeClassifier(criterion='entropy').fit(X,y).predict(X)),6)
    def test_constant(self):
        t=DecisionTreeClassifier().fit(np.ones((5,2)),np.array([1,1,0,0,1]));self.assertTrue(np.all(t.predict(np.ones((3,2)))==1))
    def test_ties(self):
        t=DecisionTreeClassifier(max_depth=0).fit(X,np.array([1,0,1,0,1,0]));self.assertEqual(t.predict(X)[0],0)
    def test_max_depth_zero(self): self.assertEqual(len(np.unique(DecisionTreeClassifier(max_depth=0).fit(X,y).predict(X))),1)
    def test_feature_mismatch(self):
        with self.assertRaises(ValueError): DecisionTreeClassifier().fit(X,y).predict(np.ones((2,3)))
    def test_y_mismatch(self):
        with self.assertRaises(ValueError): DecisionTreeClassifier().fit(X,y[:2])
    def test_predict_before_train(self):
        with self.assertRaises(RuntimeError): DecisionTreeClassifier().predict(X)
    def test_invalid_buffer(self):
        with self.assertRaises(ValueError): DecisionTreeClassifier(buffer_size=0)
    def test_invalid_criterion(self):
        with self.assertRaises(ValueError): DecisionTreeClassifier(criterion='wrong')
    def test_multiclass(self):
        t=DecisionTreeClassifier().fit(X,np.array([0,0,1,1,2,2]));self.assertEqual(len(t.predict(X)),6)
    def test_retrain_new_classes(self):
        t=DecisionTreeClassifier(buffer_size=3).partial_fit(X[:3],y[:3]);t.partial_fit(X[3:],y[3:]);self.assertTrue(np.all(t.predict(X)==1))
    def test_infinity_rejected(self):
        with self.assertRaises(ValueError): DecisionTreeClassifier().fit(np.array([[np.inf],[1]]),[0,1])
class TestPreprocessing(unittest.TestCase):
    def test_scaler_constant(self): self.assertTrue(np.isfinite(StandardScaler().fit_transform(np.ones((4,2)))).all())
    def test_scaler_stream(self):
        s=StandardScaler().partial_fit(X[:3]).partial_fit(X[3:]);np.testing.assert_allclose(s.mean_,X.mean(axis=0))
    def test_scaler_var(self):
        s=StandardScaler().partial_fit(X[:2]).partial_fit(X[2:]);np.testing.assert_allclose(s.var_,X.var(axis=0))
    def test_scaler_missing(self): self.assertEqual(StandardScaler().fit(np.array([[np.nan],[2.]])).mean_[0],2.)
    def test_imputer(self): self.assertEqual(Imputer().fit_transform(np.array([[1.],[np.nan]]))[1,0],1.)
    def test_imputer_stream(self):
        i=Imputer().partial_fit([[1.],[np.nan]]).partial_fit([[3.]]);self.assertEqual(i.statistics_[0],2.)
    def test_imputer_all_missing(self): self.assertEqual(Imputer().fit_transform([[np.nan]])[0,0],0.)
    def test_encoder_expands(self):
        e=OneHotEncoder().partial_fit([['a']]);self.assertEqual(e.transform([['a']]).shape[1],1)
        e.partial_fit([['b']]);self.assertEqual(e.transform([['a']]).shape[1],2)
    def test_encoder_unknown(self): self.assertEqual(OneHotEncoder().fit([['a']]).transform([['b']]).sum(),0)
    def test_pipeline(self):
        p=Pipeline([('impute',Imputer()),('scale',StandardScaler()),('tree',DecisionTreeClassifier(buffer_size=3))]);p.partial_fit(X[:3],y[:3]);self.assertEqual(len(p.predict(X)),6)
class TestMetrics(unittest.TestCase):
    def test_accuracy(self): self.assertEqual(StreamingMetrics().update([1,0],[1,0]).result()['accuracy'],1.)
    def test_precision(self): self.assertEqual(StreamingMetrics().update([1,0],[1,1]).result()['precision'],.5)
    def test_recall(self): self.assertEqual(StreamingMetrics().update([1,1],[1,0]).result()['recall'],.5)
    def test_f1(self): self.assertEqual(StreamingMetrics().update([1,0],[1,0]).result()['f1'],1.)
    def test_confusion(self): self.assertEqual(StreamingMetrics().update([1,0],[0,1]).result()['confusion_matrix'].sum(),2)
    def test_reset(self): self.assertEqual(StreamingMetrics().update([1],[1]).reset().result()['count'],0)
    def test_window(self): self.assertEqual(StreamingMetrics(window=2).update([1,1,0],[1,0,0]).result()['count'],2)
    def test_auc(self): self.assertAlmostEqual(StreamingMetrics().update([0,1],[0,1],[.1,.9]).result()['auc'],1.)
    def test_auc_ties(self): self.assertAlmostEqual(StreamingMetrics().update([0,1],[0,1],[.5,.5]).result()['auc'],.5)
    def test_empty(self): self.assertEqual(StreamingMetrics().result()['count'],0)
class TestStatsAndIO(unittest.TestCase):
    def test_stats_mean(self): self.assertEqual(StreamingStats().update_stats([1,2]).update_stats([3]).result()['mean'],2.)
    def test_stats_variance(self): self.assertAlmostEqual(StreamingStats().update([1,2,3]).result()['variance'],2/3)
    def test_stats_nan(self): self.assertEqual(StreamingStats().update([np.nan,2]).result()['count'],1)
    def test_quantile(self): self.assertEqual(StreamingStats().update([1,3]).quantile(.5),2.)
    def test_histogram(self): self.assertEqual(StreamingStats().update([1,2,3]).histogram()[0].sum(),3)
    def test_stream_trainer(self):
        tr=StreamTrainer(DecisionTreeClassifier(buffer_size=3));tr.fit_chunk(X[:3],y[:3]);self.assertEqual(tr.history[0]['retrain_count'],1)
    def test_csv_chunks(self):
        with tempfile.NamedTemporaryFile(mode='w',suffix='.csv',delete=False) as f:
            f.write('1,2\n3,4\n5,6\n');name=f.name
        try: self.assertEqual([len(c) for c in iter_csv_chunks(name,2)],[2,1])
        finally: os.unlink(name)
    def test_csv_read(self):
        with tempfile.NamedTemporaryFile(mode='w',suffix='.csv',delete=False) as f:
            f.write('1,2\n3,4\n');name=f.name
        try: self.assertEqual(read_csv(name).shape,(2,2))
        finally: os.unlink(name)
if __name__=='__main__': unittest.main()
