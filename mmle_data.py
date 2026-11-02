"""Strict CSV import and reproducible experiment exports (no regeneration)."""
from io import BytesIO
import csv
import json
import zipfile
import numpy as np
import pandas as pd
from mmle_engine import Dataset,jsonable


def load_csv_dataset(response_bytes,administration_bytes=None,bank_bytes=None,model='2PL'):
    header=next(csv.reader(response_bytes.decode('utf-8-sig').splitlines()))
    if len(set(header))!=len(header):
        raise ValueError('Duplicate CSV columns: represent each shared global item ID once.')
    frame=pd.read_csv(BytesIO(response_bytes))
    groups=frame.pop('group').astype(str) if 'group' in frame else pd.Series(['G1']*len(frame))
    names=list(pd.unique(groups));group=groups.map({name:i for i,name in enumerate(names)}).to_numpy()
    ids=frame.columns.tolist();y=frame.apply(pd.to_numeric,errors='raise').to_numpy(float)
    admin=np.ones_like(y,bool)
    if administration_bytes is not None:
        af=pd.read_csv(BytesIO(administration_bytes))
        if af.columns.tolist()!=ids or af.shape!=frame.shape or not np.isin(af.to_numpy(),[0,1]).all():
            raise ValueError('Administration CSV needs identical item columns / rows and only 0/1 values.')
        admin=af.to_numpy(bool)
    bank=np.full((len(ids),3),np.nan)
    if bank_bytes is not None:
        bf=pd.read_csv(BytesIO(bank_bytes))
        if 'model' in bf and not bf.model.isin([model,model+'M']).all():
            raise ValueError('Metadata models must match the fitted model.')
        if 'cats' in bf and not (bf.cats==2).all():
            raise ValueError('Only binary items with two categories are supported.')
        for item,rows in bf.groupby('id',sort=False):
            if item not in ids: raise ValueError(f'Unknown bank item {item}')
            values=rows[['a','b','c']].to_numpy(float)
            locks=rows.fixed.astype(str).str.lower().isin(['true','1']).to_numpy() if 'fixed' in rows else np.ones(len(rows),bool)
            if len(rows)>1 and (not np.allclose(values,values[0],equal_nan=True) or np.any(locks!=locks[0])):
                raise ValueError(f'Conflicting same-ID bank coefficients or locks for {item}')
            if locks[0]: bank[ids.index(item)]=values[0]
    data=Dataset(y,admin,group,ids,names,bank,generation=dict(source='Upload'))
    data.validate()
    return data


def export_experiment(data,result=None,report=None):
    output=BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        def write(name,value):
            z.writestr(name,value.to_csv(index=False) if isinstance(value,pd.DataFrame) else json.dumps(jsonable(value),indent=2,allow_nan=False))
        frame=pd.DataFrame(data.responses,columns=data.item_ids)
        frame.insert(0,'group',[data.group_names[g] for g in data.group])
        write('responses.csv',frame)
        write('administration.csv',pd.DataFrame(data.administered.astype(int),columns=data.item_ids))
        write('bank.csv',pd.DataFrame(dict(id=data.item_ids,fixed=data.fixed,a=data.bank[:,0],b=data.bank[:,1],c=data.bank[:,2])))
        write('generation.json',data.generation)
        if data.truth is not None:
            truth=pd.DataFrame(data.truth,columns=['a','b','c']);truth.insert(0,'Item',data.item_ids)
            write('truth.csv',truth)
            write('true_abilities.csv',pd.DataFrame(dict(group=data.group,theta=data.theta)))
        if result is not None:
            from dataclasses import asdict
            write('settings.json',dict(config=asdict(result.config),data_id=result.data_id,run_id=result.run_id,
                                       constraints_id=result.constraints_id,versions=result.versions,stop_reason=result.stop_reason))
            items=pd.DataFrame(result.params,columns=['a','b','c']);items.insert(0,'id',data.item_ids);items['Bank fixed']=data.fixed
            write('items.csv',items)
            write('cycles.csv',pd.DataFrame([dict(cycle=c.index,raw_loglik=c.loglik,penalized_objective=c.penalized,
                                                 item_change=c.item_change,weight_change=c.weight_change,moment_change=c.moment_change,
                                                 optimizer_OK=all(s['success'] for s in c.optimizer)) for c in result.cycles]))
            write('history.json',dict(nodes=result.nodes,initial_params=result.initial_params,initial_weights=result.initial_weights,cycles=result.cycles))
        write('validation.json',report or dict(status='Not run',executed=[],pending=['irtQ endpoint agreement','Constraints / arithmetic','Sensitivity','Replicated recovery']))
    return output.getvalue()
