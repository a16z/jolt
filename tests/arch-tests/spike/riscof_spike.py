import os
import re
import shutil
import subprocess
import shlex
import logging
import random
import string
from string import Template
import sys

import riscof.utils as utils
import riscof.constants as constants
from riscof.pluginTemplate import pluginTemplate

logger = logging.getLogger()

class spike(pluginTemplate):
    __model__ = "spike"

    #TODO: please update the below to indicate family, version, etc of your DUT.
    __version__ = "XXX"

    def __init__(self, *args, **kwargs):
        sclass = super().__init__(*args, **kwargs)

        config = kwargs.get('config')

        self.ref_exe = os.path.join(config['PATH'] if 'PATH' in config else "","spike")
        self.num_jobs = str(config['jobs'] if 'jobs' in config else 1)
        self.pluginpath=os.path.abspath(config['pluginpath'])
        self.isa_spec = os.path.abspath(config['ispec']) if 'ispec' in config else ''
        self.platform_spec = os.path.abspath(config['pspec']) if 'ispec' in config else ''
        self.make = config['make'] if 'make' in config else 'make'
        logger.debug("spike plugin initialised using the following configuration.")
        for entry in config:
            logger.debug(entry+' : '+config[entry])
        return sclass

    def initialise(self, suite, work_dir, archtest_env):
        self.suite = suite
        if shutil.which(self.ref_exe) is None:
            logger.error(f'Executable not found: {self.ref_exe}. Please install it or check your PATH to proceed further.')
            raise SystemExit(1)
        self.work_dir = work_dir

        #TODO: The following assumes you are using the riscv-gcc toolchain. If
        #      not please change appropriately
        self.objdump_cmd = 'riscv64-unknown-elf-objdump -D {0} > {2};'
        self.compile_cmd = 'riscv64-unknown-elf-gcc -march={0} \
         -static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles\
         -T '+self.pluginpath+'/env/link.ld\
         -I '+self.pluginpath+'/env/\
         -I ' + archtest_env

        # set all the necessary variables like compile command, elf2hex
        # commands, objdump cmds. etc whichever you feel necessary and required
        # for your plugin.

    def build(self, isa_yaml, platform_yaml):
        ispec = utils.load_yaml(isa_yaml)['hart0']
        self.xlen = ('64' if 64 in ispec['supported_xlen'] else '32')
        self.isa = 'rv' + self.xlen
        #TODO: The following assumes you are using the riscv-gcc toolchain. If
        #      not please change appropriately
        self.compile_cmd = self.compile_cmd+' -mabi='+('lp64 ' if 64 in ispec['supported_xlen'] else 'ilp32 ')
        # RISC-V standard canonical order: I, E, M, A, F, D, Q, C, V, Z extensions
        if "I" in ispec["ISA"]:
            self.isa += 'i'
        if "E" in ispec["ISA"]:
            self.isa += 'e'
        if "M" in ispec["ISA"]:
            self.isa += 'm'
        if "A" in ispec["ISA"]:
            self.isa += 'a'
        if "F" in ispec["ISA"]:
            self.isa += 'f'
        if "D" in ispec["ISA"]:
            self.isa += 'd'
        if "Q" in ispec["ISA"]:
            self.isa += 'q'
        if "C" in ispec["ISA"]:
            self.isa += 'c'
        if "V" in ispec["ISA"]:
            self.isa += 'v'
        
        # Z extensions (alphabetical order)
        if "Zba" in ispec["ISA"]:
            self.isa += '_Zba'
        if "Zbb" in ispec["ISA"]:
            self.isa += '_Zbb'
        if "Zbc" in ispec["ISA"]:
            self.isa += '_Zbc'
        if "Zbkb" in ispec["ISA"]:
            self.isa += '_Zbkb'
        if "Zbkc" in ispec["ISA"]:
            self.isa += '_Zbkc'
        if "Zbkx" in ispec["ISA"]:
            self.isa += '_Zbkx'
        if "Zbs" in ispec["ISA"]:
            self.isa += '_Zbs'
        if "Zca" in ispec["ISA"]:
            self.isa += '_Zca'
        if "Zcb" in ispec["ISA"]:
            self.isa += '_Zcb'
        if "Zcmop" in ispec["ISA"]:
            self.isa += '_Zcmop'
        if "Zfa" in ispec["ISA"]:
            self.isa += '_Zfa'
        if "Zfh" in ispec["ISA"]:
            self.isa += '_Zfh'
        if "Zicboz" in ispec["ISA"]:
            self.isa += '_Zicboz'
        if "Zicond" in ispec["ISA"]:
            self.isa += '_Zicond'
        if "Zicsr" in ispec["ISA"]:
            self.isa += '_Zicsr'
        if "Zimop" in ispec["ISA"]:
            self.isa += '_Zimop'
        if "Zknd" in ispec["ISA"]:
            self.isa += '_Zknd'
        if "Zkne" in ispec["ISA"]:
            self.isa += '_Zkne'
        if "Zknh" in ispec["ISA"]:
            self.isa += '_Zknh'
        if "Zksed" in ispec["ISA"]:
            self.isa += '_Zksed'
        if "Zksh" in ispec["ISA"]:
            self.isa += '_Zksh'

        # based on the validated isa and platform configure your simulator or
        # build your RTL here

    def runTests(self, testList, cgf_file=None):
        if os.path.exists(self.work_dir+ "/Makefile." + self.name[:-1]):
            os.remove(self.work_dir+ "/Makefile." + self.name[:-1])
        make = utils.makeUtil(makefilePath=os.path.join(self.work_dir, "Makefile." + self.name[:-1]))
        make.makeCommand = self.make + ' -j' + self.num_jobs
        for file in testList:
            testentry = testList[file]
            test = testentry['test_path']
            test_dir = testentry['work_dir']
            test_name = test.rsplit('/',1)[1][:-2]

            elf = 'ref.elf'

            execute = "@cd "+testentry['work_dir']+";"

            cmd = self.compile_cmd.format(testentry['isa'].lower(), self.xlen) + ' ' + test + ' -o ' + elf

            #TODO: we are using -D to enable compile time macros. If your
            #      toolchain is not riscv-gcc you may want to change the below code
            compile_cmd = cmd + ' -D' + " -D".join(testentry['macros'])
            execute+=compile_cmd+";"

            execute += self.objdump_cmd.format(elf, self.xlen, 'ref.disass')
            sig_file = os.path.join(test_dir, self.name[:-1] + ".signature")

            #TODO: You will need to add any other arguments to your DUT
            #      executable if any in the quotes below
            execute += self.ref_exe + ' --instructions=10000000 --misaligned --isa={0} +signature={1} +signature-granularity=4 {2}'.format(self.isa, sig_file, elf)

            #TODO: The following is useful only if your reference model can
            #      support coverage extraction from riscv-isac. Else leave it
            #      commented out

            #cov_str = ' '
            #for label in testentry['coverage_labels']:
            #    cov_str+=' -l '+label
            #if cgf_file is not None:
            #    coverage_cmd = 'riscv_isac --verbose info coverage -d \
            #            -t {0}.log --parser-name c_sail -o coverage.rpt  \
            #            --sig-label begin_signature  end_signature \
            #            --test-label rvtest_code_begin rvtest_code_end \
            #            -e {0}.elf -c {1} -x{2} {3};'.format(\
            #            test_name, ' -c '.join(cgf_file), self.xlen, cov_str)
            #else:
            #    coverage_cmd = ''
            #execute+=coverage_cmd

            make.add_target(execute)
        make.execute_all(self.work_dir)
