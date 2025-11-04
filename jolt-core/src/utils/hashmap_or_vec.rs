use allocative::Allocative;
use itertools::Either;
use std::{
    collections::HashMap,
    convert::identity,
    ops::{Index, IndexMut},
};

#[derive(Clone, Allocative)]
pub enum HashMapOrVec<T: Clone + Default> {
    HashMap(HashMap<usize, T>),
    Vec(Vec<Option<T>>),
}

impl<T: Clone + Default + Send + std::fmt::Debug> HashMapOrVec<T> {
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
            HashMapOrVec::HashMap(map) => {
                if !map.contains_key(&k) {
                    map.insert(k, v);
                    Ok(())
                } else {
                    Err(&map[&k])
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
            HashMapOrVec::HashMap(map) => map.shrink_to_fit(),
            HashMapOrVec::Vec(vec) => vec.shrink_to_fit(),
        };
    }

    pub fn get(&self, index: usize) -> Option<T> {
        match self {
            HashMapOrVec::HashMap(map) => map.get(&index).cloned(),
            HashMapOrVec::Vec(vec) => vec[index].clone(),
        }
    }

    pub fn clear(&mut self) {
        match self {
            HashMapOrVec::HashMap(map) => map.clear(),
            HashMapOrVec::Vec(vec) => vec.fill(None),
        };
    }

    pub fn clone_from(&mut self, other: &Self) {
        match (self, other) {
            (HashMapOrVec::HashMap(map), HashMapOrVec::HashMap(other)) => {
                map.clear();
                map.extend(other.iter().map(|(k, v)| (*k, v.clone())));
            }
            (HashMapOrVec::Vec(vec), HashMapOrVec::Vec(other)) => {
                vec.clone_from_slice(&other);
            }
            _ => panic!("mismatched data structures"),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        match self {
            HashMapOrVec::HashMap(map) => Either::Left(map.values()),
            HashMapOrVec::Vec(vec) => Either::Right(vec.iter().filter_map(|x| x.as_ref())),
        }
    }

    pub fn drain(&mut self) -> impl Iterator<Item = T> + use<'_, T> {
        match self {
            HashMapOrVec::HashMap(map) => Either::Left(map.drain().map(|(_k, v)| v)),
            HashMapOrVec::Vec(vec) => Either::Right(vec.drain(..).filter_map(identity)),
        }
    }
}

impl<T: Clone + Default + Send> Index<usize> for HashMapOrVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            HashMapOrVec::HashMap(map) => map
                .get(&index)
                .as_ref()
                .unwrap_or_else(|| panic!("No entry for key {index}")),
            HashMapOrVec::Vec(vec) => vec[index]
                .as_ref()
                .unwrap_or_else(|| panic!("No entry for key {index}")),
        }
    }
}

impl<T: Clone + Default + Send> IndexMut<usize> for HashMapOrVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            HashMapOrVec::HashMap(map) => map.entry(index).or_insert(T::default()),
            HashMapOrVec::Vec(vec) => vec[index].get_or_insert_default(),
        }
    }
}
