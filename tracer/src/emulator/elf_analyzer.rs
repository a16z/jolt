extern crate fnv;

#[cfg(feature = "std")]
use self::fnv::FnvHashMap;
#[cfg(not(feature = "std"))]
use alloc::collections::btree_map::BTreeMap as FnvHashMap;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

pub struct Header {
    pub e_width: u8, // 32 or 64
    _e_class: u8,
    _e_endian: u8,
    _e_elf_version: u8,
    _e_osabi: u8,
    _e_abi_version: u8,
    _e_type: u16,
    _e_machine: u16,
    _e_version: u32,
    pub e_entry: u64,
    _e_phoff: u64,
    e_shoff: u64,
    _e_flags: u32,
    _e_ehsize: u16,
    _e_phentsize: u16,
    _e_phnum: u16,
    _e_shentsize: u16,
    e_shnum: u16,
    _e_shstrndx: u16,
}

#[derive(Debug)]
pub struct SectionHeader {
    #[allow(dead_code)]
    sh_name: u32,
    pub sh_type: u32,
    pub sh_flags: u64,
    pub sh_addr: u64,
    pub sh_offset: u64,
    pub sh_size: u64,
    pub sh_link: u32,
    _sh_info: u32,
    _sh_addralign: u64,
    _sh_entsize: u64,
}

pub struct SymbolEntry {
    st_name: u32,
    st_info: u8,
    _st_other: u8,
    _st_shndx: u16,
    st_value: u64,
    _st_size: u64,
}

pub struct ElfAnalyzer {
    data: Vec<u8>,
}

impl ElfAnalyzer {
    /// Creates a new `ElfAnalyzer`.
    ///
    /// # Arguments
    /// * `data` ELF file content binary
    pub fn new(data: &[u8]) -> Self {
        ElfAnalyzer {
            data: data.to_vec(),
        }
    }

    pub fn validate(&self) -> bool {
        if self.data.len() < 4
            || self.data[0] != 0x7f
            || self.data[1] != 0x45
            || self.data[2] != 0x4c
            || self.data[3] != 0x46
        {
            return false;
        }
        true
    }

    pub fn read_header(&self) -> Header {
        let e_class = self.read_byte(4);

        let e_width = match e_class {
            1 => 32,
            2 => 64,
            _ => panic!("Unknown e_class:{e_class:X}"),
        };

        let e_endian = self.read_byte(5);
        let e_elf_version = self.read_byte(6);
        let e_osabi = self.read_byte(7);
        let e_abi_version = self.read_byte(8);

        let mut offset = 0x10;

        let e_type = self.read_halfword(offset);
        offset += 2;

        let e_machine = self.read_halfword(offset);
        offset += 2;

        let e_version = self.read_word(offset);
        offset += 4;

        let e_entry = match e_width {
            64 => {
                let data = self.read_doubleword(offset);
                offset += 8;
                data
            }
            _ => {
                let data = self.read_word(offset);
                offset += 4;
                data as u64
            }
        };

        let e_phoff = match e_width {
            64 => {
                let data = self.read_doubleword(offset);
                offset += 8;
                data
            }
            _ => {
                let data = self.read_word(offset);
                offset += 4;
                data as u64
            }
        };

        let e_shoff = match e_width {
            64 => {
                let data = self.read_doubleword(offset);
                offset += 8;
                data
            }
            _ => {
                let data = self.read_word(offset);
                offset += 4;
                data as u64
            }
        };

        let e_flags = self.read_word(offset);
        offset += 4;

        let e_ehsize = self.read_halfword(offset);
        offset += 2;

        let e_phentsize = self.read_halfword(offset);
        offset += 2;

        let e_phnum = self.read_halfword(offset);
        offset += 2;

        let e_shentsize = self.read_halfword(offset);
        offset += 2;

        let e_shnum = self.read_halfword(offset);
        offset += 2;

        let e_shstrndx = self.read_halfword(offset);

        Header {
            e_width,
            _e_class: e_class,
            _e_endian: e_endian,
            _e_elf_version: e_elf_version,
            _e_osabi: e_osabi,
            _e_abi_version: e_abi_version,
            _e_type: e_type,
            _e_machine: e_machine,
            _e_version: e_version,
            e_entry,
            _e_phoff: e_phoff,
            e_shoff,
            _e_flags: e_flags,
            _e_ehsize: e_ehsize,
            _e_phentsize: e_phentsize,
            _e_phnum: e_phnum,
            _e_shentsize: e_shentsize,
            e_shnum,
            _e_shstrndx: e_shstrndx,
        }
    }

    pub fn read_section_headers(&self, header: &Header) -> Vec<SectionHeader> {
        let mut headers = Vec::new();
        let mut offset = header.e_shoff as usize;
        for _i in 0..header.e_shnum {
            let sh_name = self.read_word(offset);
            offset += 4;

            let sh_type = self.read_word(offset);
            offset += 4;

            let sh_flags = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let sh_addr = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let sh_offset = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let sh_size = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let sh_link = self.read_word(offset);
            offset += 4;

            let sh_info = self.read_word(offset);
            offset += 4;

            let sh_addralign = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            let sh_entsize = match header.e_width {
                64 => {
                    let data = self.read_doubleword(offset);
                    offset += 8;
                    data
                }
                32 => {
                    let data = self.read_word(offset);
                    offset += 4;
                    data as u64
                }
                _ => panic!("Not happen"),
            };

            headers.push(SectionHeader {
                sh_name,
                sh_type,
                sh_flags,
                sh_addr,
                sh_offset,
                sh_size,
                sh_link,
                _sh_info: sh_info,
                _sh_addralign: sh_addralign,
                _sh_entsize: sh_entsize,
            });
        }

        headers
    }

    pub fn read_symbol_entries(
        &self,
        header: &Header,
        symbol_table_section_headers: &[&SectionHeader],
    ) -> Vec<SymbolEntry> {
        let mut entries = Vec::new();
        for section_header in symbol_table_section_headers {
            let sh_offset = section_header.sh_offset;
            let sh_size = section_header.sh_size;

            let mut offset = sh_offset as usize;

            let entry_size = match header.e_width {
                64 => 24,
                32 => 16,
                _ => panic!("Not happen"),
            };

            for _j in 0..(sh_size / entry_size) {
                let st_name;
                let st_info;
                let _st_other;
                let _st_shndx;
                let st_value;
                let _st_size;

                match header.e_width {
                    64 => {
                        st_name = self.read_word(offset);
                        offset += 4;

                        st_info = self.read_byte(offset);
                        offset += 1;

                        _st_other = self.read_byte(offset);
                        offset += 1;

                        _st_shndx = self.read_halfword(offset);
                        offset += 2;

                        st_value = self.read_doubleword(offset);
                        offset += 8;

                        _st_size = self.read_doubleword(offset);
                        offset += 8;
                    }
                    32 => {
                        st_name = self.read_word(offset);
                        offset += 4;

                        st_value = self.read_word(offset) as u64;
                        offset += 4;

                        _st_size = self.read_word(offset) as u64;
                        offset += 4;

                        st_info = self.read_byte(offset);
                        offset += 1;

                        _st_other = self.read_byte(offset);
                        offset += 1;

                        _st_shndx = self.read_halfword(offset);
                        offset += 2;
                    }
                    _ => panic!("No happen"),
                };

                entries.push(SymbolEntry {
                    st_name,
                    st_info,
                    _st_other,
                    _st_shndx,
                    st_value,
                    _st_size,
                });
            }
        }
        entries
    }

    /// Builds the symbol name -> address map of every symbol table section.
    ///
    /// Each symbol table resolves its names through the string table named by
    /// its own `sh_link`, as the ELF spec requires; picking the first
    /// `SHT_STRTAB` section instead can select `.shstrtab` (LLD emits it
    /// before `.strtab`).
    pub fn read_symbol_map(
        &self,
        header: &Header,
        section_headers: &[SectionHeader],
    ) -> FnvHashMap<String, u64> {
        let mut map = FnvHashMap::default();
        for symbol_table in section_headers.iter().filter(|s| s.sh_type == 2) {
            let Some(string_table) = section_headers.get(symbol_table.sh_link as usize) else {
                continue;
            };
            let entries = self.read_symbol_entries(header, &[symbol_table]);
            map.extend(self.create_symbol_map(&entries, string_table));
        }
        map
    }

    fn read_strings(&self, section_header: &SectionHeader, index: u64) -> String {
        let sh_offset = section_header.sh_offset;
        let sh_size = section_header.sh_size;
        let mut pos = 0;
        let mut symbol = String::new();
        loop {
            let addr = sh_offset + index + pos;
            if addr >= sh_offset + sh_size {
                break;
            }
            let value = self.read_byte(addr as usize);
            if value == 0 {
                break;
            }
            symbol.push(value as char);
            pos += 1;
        }
        symbol
    }

    /// Creates a symbol - virtual address mapping from symbol entries
    /// and a string table section.
    ///
    /// # Arguments
    /// * `entries` Symbol entries
    /// * `string_table_section_header` The header of the string table section
    pub fn create_symbol_map(
        &self,
        entries: &Vec<SymbolEntry>,
        string_table_section_header: &SectionHeader,
    ) -> FnvHashMap<String, u64> {
        let mut map = FnvHashMap::default();
        for entry in entries {
            let st_info = entry.st_info;
            let st_name = entry.st_name;
            let st_value = entry.st_value;

            if (st_info & 0x2) != 0x2 && (st_info & 0xf) != 0 {
                continue;
            }

            let symbol = self.read_strings(string_table_section_header, st_name as u64);

            if !symbol.is_empty() {
                map.insert(symbol, st_value);
            }
        }
        map
    }

    pub fn read_byte(&self, offset: usize) -> u8 {
        self.data[offset]
    }

    fn read_halfword(&self, offset: usize) -> u16 {
        let mut data = 0;
        for i in 0..2 {
            data |= (self.read_byte(offset + i) as u16) << (8 * i);
        }
        data
    }

    fn read_word(&self, offset: usize) -> u32 {
        let mut data = 0;
        for i in 0..4 {
            data |= (self.read_byte(offset + i) as u32) << (8 * i);
        }
        data
    }

    fn read_doubleword(&self, offset: usize) -> u64 {
        let mut data = 0;
        for i in 0..8 {
            data |= (self.read_byte(offset + i) as u64) << (8 * i);
        }
        data
    }
}

/// Hand-assembled ELF64 fixtures for emulator tests. Field offsets follow the
/// System V gABI ELF64 layout, so these bytes are an oracle independent of the
/// analyzer under test.
#[cfg(any(test, feature = "test-utils"))]
pub mod test_elf {
    pub struct TestSymbol {
        pub name: &'static str,
        pub value: u64,
        /// st_info (binding << 4 | type), e.g. 0x10 = GLOBAL|NOTYPE, 0x12 = GLOBAL|FUNC
        pub info: u8,
        pub size: u64,
    }

    /// Order of the two `SHT_STRTAB` sections in the section header table.
    #[derive(Clone, Copy)]
    pub enum StrtabOrder {
        /// `.strtab` before `.shstrtab`, as GNU ld emits them.
        GnuLd,
        /// `.shstrtab` before `.strtab`, as LLD emits them.
        Lld,
    }

    /// Builds a minimal but well-formed RV64 ELF: `.text` loaded at
    /// 0x8000_0000 with the given instruction words, plus a symbol table whose
    /// `sh_link` names `.strtab` in either section order.
    pub fn build_elf64(text: &[u32], symbols: &[TestSymbol], order: StrtabOrder) -> Vec<u8> {
        const TEXT_ADDR: u64 = 0x8000_0000;
        let text_bytes: Vec<u8> = text.iter().flat_map(|w| w.to_le_bytes()).collect();

        let align8 = |offset: usize| offset.div_ceil(8) * 8;

        let text_offset = 0x40; // right after the 64-byte ELF header
        let symtab_offset = align8(text_offset + text_bytes.len());
        let symtab_size = 24 * (symbols.len() + 1); // null entry + symbols

        // .strtab: leading NUL, then NUL-terminated names
        let mut strtab = vec![0u8];
        let mut name_offsets = Vec::new();
        for symbol in symbols {
            name_offsets.push(strtab.len() as u32);
            strtab.extend_from_slice(symbol.name.as_bytes());
            strtab.push(0);
        }
        let strtab_offset = symtab_offset + symtab_size;

        let shstrtab: &[u8] = b"\0.text\0.symtab\0.strtab\0.shstrtab\0";
        let shstrtab_offset = strtab_offset + strtab.len();
        let shoff = align8(shstrtab_offset + shstrtab.len());

        // Section header indices of .strtab and .shstrtab.
        let (strtab_index, shstrtab_index): (u32, u16) = match order {
            StrtabOrder::GnuLd => (3, 4),
            StrtabOrder::Lld => (4, 3),
        };

        let mut elf = Vec::new();
        // ELF header
        elf.extend_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        elf.extend_from_slice(&2u16.to_le_bytes()); // e_type = EXEC
        elf.extend_from_slice(&0xf3u16.to_le_bytes()); // e_machine = RISC-V
        elf.extend_from_slice(&1u32.to_le_bytes()); // e_version
        elf.extend_from_slice(&TEXT_ADDR.to_le_bytes()); // e_entry
        elf.extend_from_slice(&0u64.to_le_bytes()); // e_phoff
        elf.extend_from_slice(&(shoff as u64).to_le_bytes()); // e_shoff
        elf.extend_from_slice(&0u32.to_le_bytes()); // e_flags
        elf.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
        elf.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
        elf.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
        elf.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
        elf.extend_from_slice(&5u16.to_le_bytes()); // e_shnum
        elf.extend_from_slice(&shstrtab_index.to_le_bytes()); // e_shstrndx
        assert_eq!(elf.len(), 0x40);

        // .text content
        elf.extend_from_slice(&text_bytes);
        elf.resize(symtab_offset, 0);

        // .symtab: null entry then the given symbols (st_shndx = .text)
        elf.extend_from_slice(&[0u8; 24]);
        for (symbol, name_offset) in symbols.iter().zip(&name_offsets) {
            elf.extend_from_slice(&name_offset.to_le_bytes());
            elf.push(symbol.info);
            elf.push(0); // st_other
            elf.extend_from_slice(&1u16.to_le_bytes()); // st_shndx
            elf.extend_from_slice(&symbol.value.to_le_bytes());
            elf.extend_from_slice(&symbol.size.to_le_bytes());
        }

        elf.extend_from_slice(&strtab);
        elf.extend_from_slice(shstrtab);
        elf.resize(shoff, 0);

        let mut push_shdr = |name: u32,
                             sh_type: u32,
                             flags: u64,
                             addr: u64,
                             offset: u64,
                             size: u64,
                             link: u32,
                             info: u32,
                             addralign: u64,
                             entsize: u64| {
            let elf = &mut elf;
            elf.extend_from_slice(&name.to_le_bytes());
            elf.extend_from_slice(&sh_type.to_le_bytes());
            elf.extend_from_slice(&flags.to_le_bytes());
            elf.extend_from_slice(&addr.to_le_bytes());
            elf.extend_from_slice(&offset.to_le_bytes());
            elf.extend_from_slice(&size.to_le_bytes());
            elf.extend_from_slice(&link.to_le_bytes());
            elf.extend_from_slice(&info.to_le_bytes());
            elf.extend_from_slice(&addralign.to_le_bytes());
            elf.extend_from_slice(&entsize.to_le_bytes());
        };

        push_shdr(0, 0, 0, 0, 0, 0, 0, 0, 0, 0); // SHT_NULL
        push_shdr(
            1,   // ".text"
            1,   // SHT_PROGBITS
            0x6, // ALLOC | EXECINSTR
            TEXT_ADDR,
            text_offset as u64,
            text_bytes.len() as u64,
            0,
            0,
            4,
            0,
        );
        push_shdr(
            7, // ".symtab"
            2, // SHT_SYMTAB
            0,
            0,
            symtab_offset as u64,
            symtab_size as u64,
            strtab_index, // link to .strtab
            1,            // one local symbol (the null entry)
            8,
            24,
        );
        let strtab_shdr = (15, strtab_offset, strtab.len()); // ".strtab"
        let shstrtab_shdr = (23, shstrtab_offset, shstrtab.len()); // ".shstrtab"
        let string_tables = match order {
            StrtabOrder::GnuLd => [strtab_shdr, shstrtab_shdr],
            StrtabOrder::Lld => [shstrtab_shdr, strtab_shdr],
        };
        for (name, offset, size) in string_tables {
            push_shdr(name, 3, 0, 0, offset as u64, size as u64, 0, 0, 1, 0); // SHT_STRTAB
        }

        elf
    }
}
