"""Memory-efficient metadata TF-IDF and personalized content-based scoring.

The vocabulary/IDF corpus is made only from distinct programs in the historical
training schedule. Candidate metadata can be transformed without refitting the
vocabulary. No user viewing information is read by this module.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
import unicodedata

import numpy as np
from scipy.sparse import csr_matrix, diags

FIELDS=("category","genre","title","director","actors","year","channel")
WORD_RE=re.compile(r"[^\W_]+",re.UNICODE)
NAME_SPLIT_RE=re.compile(r"[,;|/]+")


def slug(value):
    text=unicodedata.normalize("NFKC",str(value)).casefold()
    return "_".join(WORD_RE.findall(text))


def metadata_tokens(row):
    """Return deterministic field-prefixed terms for one raw schedule row."""
    tokens={field:set() for field in FIELDS}
    def add(field,token):
        if token:tokens[field].add(field.upper()+"__"+token)
    add("category",slug(row.get("tv_show_category","")))
    for field in ("tv_show_genre_1","tv_show_genre_2","tv_show_genre_3"):
        add("genre",slug(row.get(field,"")))
    for word in WORD_RE.findall(unicodedata.normalize("NFKC",row.get("tv_show_title","")).casefold()):
        add("title",word)
    for name in NAME_SPLIT_RE.split(row.get("director", "")):
        add("director",slug(name))
    for name in NAME_SPLIT_RE.split(row.get("actors", "")):
        add("actors",slug(name))
    year=row.get("year_of_production","").strip()
    if year.isascii() and year.isdecimal() and len(year)==4:
        add("year",year)
        add("year",f"decade_{(int(year)//10)*10}s")
    channel_id=row.get("channel_id","").strip()
    if channel_id.isascii() and channel_id.isdecimal():add("channel",f"id_{int(channel_id)}")
    add("channel",slug(row.get("channel_title","")))
    return {field:values for field,values in tokens.items() if values}


def aggregate_metadata_rows(rows):
    """Aggregate repeated schedule rows into one field-token set per program."""
    result=defaultdict(lambda:defaultdict(set))
    for program_id,row in rows:
        if not program_id or program_id=="0":continue
        for field,terms in metadata_tokens(row).items():result[program_id][field].update(terms)
    return {item:{f:set(values) for f,values in fields.items()} for item,fields in result.items()}


@dataclass(frozen=True)
class FeatureSpace:
    training_items:tuple
    vocabulary:dict
    inverse_document_frequency:np.ndarray
    document_count:int
    training_matrix:csr_matrix
    document_frequency:dict

    @classmethod
    def fit(cls,training_documents,field_weights):
        items=tuple(sorted(training_documents,key=int))
        df=Counter()
        for item in items:
            terms=set().union(*(training_documents[item].get(field,set()) for field in FIELDS))
            df.update(terms)
        vocabulary={term:i for i,term in enumerate(sorted(df))}
        n=max(1,len(items))
        idf=np.asarray([math.log((1+n)/(1+df[term]))+1.0 for term in sorted(df)],dtype=np.float64)
        obj=cls(items,vocabulary,idf,len(items),csr_matrix((len(items),len(vocabulary)),dtype=np.float64),dict(df))
        matrix=obj.transform(training_documents,items,field_weights)
        return cls(items,vocabulary,idf,len(items),matrix,dict(df))

    def transform(self,documents,items,field_weights):
        rows=[];cols=[];values=[]
        for row,item in enumerate(items):
            fields=documents.get(item,{})
            counts=Counter(term for field in FIELDS for term in fields.get(field,set()))
            for term,tf in counts.items():
                col=self.vocabulary.get(term)
                field=term.split("__",1)[0].casefold()
                weight=float(field_weights.get(field,0.0))
                if col is None or weight<=0:continue
                rows.append(row);cols.append(col)
                values.append((1.0+math.log(tf))*self.inverse_document_frequency[col]*weight)
        matrix=csr_matrix((np.asarray(values,dtype=np.float64),(np.asarray(rows,dtype=np.int32),np.asarray(cols,dtype=np.int32))),
                          shape=(len(items),len(self.vocabulary)),dtype=np.float64)
        norms=np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        inv=np.zeros_like(norms);nz=norms>0;inv[nz]=1/norms[nz]
        return matrix.multiply(inv[:,None]).tocsr()


def build_user_profiles(profiles,training_items,training_item_index,training_matrix,cohort,*,weighting="log_frequency"):
    """Build L2-normalized user metadata profiles from training interactions only."""
    if weighting not in {"airings","log_frequency","watched_seconds"}:
        raise ValueError("Unknown user-profile weighting")
    row=[];col=[];data=[]
    for u_index,user in enumerate(cohort):
        profile=profiles.get(user,{})
        for item,(airings,seconds,_) in profile.items():
            item_index=training_item_index.get(item)
            if item_index is None:continue
            if weighting=="airings":weight=float(airings)
            elif weighting=="log_frequency":weight=math.log1p(airings)
            else:weight=math.log1p(seconds)
            if weight>0:row.append(u_index);col.append(item_index);data.append(weight)
    user_item=csr_matrix((np.asarray(data,dtype=np.float64),(np.asarray(row,dtype=np.int32),np.asarray(col,dtype=np.int32))),
                         shape=(len(cohort),len(training_items)),dtype=np.float64)
    vectors=(user_item@training_matrix).tocsr()
    norms=np.sqrt(np.asarray(vectors.multiply(vectors).sum(axis=1)).ravel())
    inv=np.zeros_like(norms);nz=norms>0;inv[nz]=1/norms[nz]
    return vectors.multiply(inv[:,None]).tocsr(),{"users_with_profile":int(np.count_nonzero(nz)),
        "cold_users":int(len(cohort)-np.count_nonzero(nz)),"profile_program_pairs":int(user_item.nnz),
        "profile_vector_nnz":int(vectors.nnz)}


def rank_candidates(candidate_matrix,user_profiles,candidates,fallback,*,batch_size=64):
    """Sparse cosine ranking with the shared fallback for zero/uncovered candidates."""
    fallback=tuple(fallback);pool=set(candidates);fallback_rank={item:i for i,item in enumerate(fallback)}
    output={};diagnostics={}
    for start in range(0,user_profiles.shape[0],batch_size):
        sims=(user_profiles[start:start+batch_size]@candidate_matrix.T).tocsr()
        for local in range(sims.shape[0]):
            u=start+local;left,right=sims.indptr[local:local+2]
            indices=sims.indices[left:right];values=sims.data[left:right]
            positive=[(candidates[int(i)],float(s)) for i,s in zip(indices,values) if s>0]
            positive.sort(key=lambda pair:(-pair[1],fallback_rank.get(pair[0],len(fallback)),int(pair[0])))
            ordered=[item for item,_ in positive]
            seen=set(ordered)
            full=ordered+[item for item in fallback if item in pool and item not in seen]
            output[u]=tuple(full)
            diagnostics[u]={"positive_similarity_candidates":len(positive),
                "fallback_slots_top5":max(0,5-min(5,len(positive))),"profile_is_nonempty":bool(user_profiles[u].nnz)}
    return output,diagnostics


FIELD_WEIGHT_CONFIGS={
    "category_genre_emphasis":{"category":2.5,"genre":2.0,"title":.4,"director":.2,"actors":.15,"year":.1,"channel":.05},
    "balanced_metadata":{"category":2.0,"genre":1.5,"title":1.0,"director":.5,"actors":.4,"year":.25,"channel":.15},
    "title_cast_emphasis":{"category":.75,"genre":1.0,"title":1.5,"director":1.0,"actors":1.25,"year":.2,"channel":.05},
    "channel_context_emphasis":{"category":1.5,"genre":1.25,"title":.8,"director":.3,"actors":.2,"year":.1,"channel":1.0},
}
