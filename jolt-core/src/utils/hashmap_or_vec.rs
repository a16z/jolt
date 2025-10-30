use std::{
    collections::HashMap,
    ops::{Index, IndexMut},
};

#[derive(Clone)]
pub enum HashMapOrVec<T: Clone + Default> {
    HashMap(HashMap<usize, T>),
    Vec(Vec<Option<T>>),
}

impl<T: Clone + Default + Send> HashMapOrVec<T> {
    /// Create a new HashMapOrVec with the given range and capacity.
    /// If the range is less than or equal to the capacity, use a Vec.
    /// Otherwise, use a HashMap.
    pub fn new(range: usize, capacity: usize) -> Self {
        if range <= capacity {
            HashMapOrVec::Vec(vec![None; range])
        } else {
            HashMapOrVec::HashMap(HashMap::with_capacity(capacity))
        }
    }

    /// Try to insert a value into the HashMapOrVec.
    /// If the key already exists, return an error with the existing value.
    pub fn try_insert(&mut self, k: usize, v: T) -> Result<(), &T> {
        match self {
            HashMapOrVec::HashMap(hashmap) => {
                if !hashmap.contains_key(&k) {
                    hashmap.insert(k, v);
                    Ok(())
                } else {
                    Err(&hashmap[&k])
                }
            }
            HashMapOrVec::Vec(vec) => {
                if vec[k].is_none() {
                    vec[k] = Some(v);
                    Ok(())
                } else {
                    Err(vec[k].as_ref().unwrap())
                }
            }
        }
    }

    /// Shrink the HashMapOrVec to fit its contents.
    pub fn shrink_to_fit(&mut self) {
        match self {
            HashMapOrVec::HashMap(hashmap) => hashmap.shrink_to_fit(),
            HashMapOrVec::Vec(vec) => vec.shrink_to_fit(),
        };
    }
}

impl<T: Clone + Default + Send> Index<usize> for HashMapOrVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            HashMapOrVec::HashMap(map) => &map[&index],
            HashMapOrVec::Vec(vec) => vec[index].as_ref().unwrap(),
        }
    }
}

impl<T: Clone + Default + Send> IndexMut<usize> for HashMapOrVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            HashMapOrVec::HashMap(map) => map.entry(index).or_insert(T::default()),
            HashMapOrVec::Vec(vec) => vec[index].as_mut().unwrap(),
        }
    }
}
