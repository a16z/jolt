// For jolt1

pragma circom 2.2.1;
include "./poseidon.circom";
include "./../fields/non_native/utils.circom";
include "./../fields/non_native/non_native_over_bn_base.circom";

bus Fp() {
    signal limbs[3];
}

bus G1Affine {
    Fp() x;
    Fp() y;
}

bus Transcript {
    signal state;
    signal nRounds;
}

template TranscriptNew(){
    input signal scalar;
    output Transcript() new_transcript;

    var stateSize = getStateSize();
    var capacity = getCapacity();

    // let mut hasher = Self::new() gives state = vec![K::zero(); parameters.rate + parameters.capacity];
    var int_state[stateSize] = [0, scalar, 0, 0, 0];

    component permute = permute(1);
    permute.state <-- int_state;

    new_transcript.state <== permute.finalState[capacity];
    new_transcript.nRounds <== 0;
}

template AppendScalar() {
    input signal scalar;
    input Transcript() transcript;
    output Transcript() up_transcript;

    var stateSize = getStateSize();
    var capacity = getCapacity();

    var int_state[stateSize] = [0, transcript.state, transcript.nRounds, scalar, 0];
 
    component permute = permute(1);
    permute.state <-- int_state;

    up_transcript.state <== permute.finalState[capacity];
    up_transcript.nRounds <== transcript.nRounds + 1;
}

template AppendScalars(nScalars) {
    input signal scalars[nScalars];
    input Transcript() transcript;
    output Transcript() up_transcript;

    Transcript() intAppendScalars[nScalars + 1];

    intAppendScalars[0] <== transcript;

   for (var i = 0; i < nScalars; i++)
   {
        intAppendScalars[i + 1] <== AppendScalar()(scalars[i], intAppendScalars[i]);
   }
   
    up_transcript <== intAppendScalars[nScalars];
}

template AppendBytes(size) {
    input signal bytes[size];
    input Transcript() transcript;
    output Transcript() up_transcript;

    var stateSize = getStateSize();
    var capacity = getCapacity();

    var int_state_len = 2 + size;
    var int_state[int_state_len];

    int_state[0] = transcript.state;
    int_state[1] = transcript.nRounds;

    for(var i = 0; i < size; i++) {
        int_state[2 + i] = bytes[i];
    }

    signal new_state[stateSize] <== permute_hash(int_state_len)(int_state);
   
    up_transcript.state <== new_state[capacity];
    up_transcript.nRounds <== transcript.nRounds + 1;
}

template AppendPoint(){
    input G1Affine() point;
    input Transcript() transcript;
    output Transcript() up_transcript;

    var stateSize = getStateSize();
    var capacity = getCapacity();

    var int_state[stateSize] = [0, transcript.state, transcript.nRounds, point.x.limbs[0], point.x.limbs[1]];

    component permute = permute(1);
    permute.state <-- int_state; 
     
    var int_state_y[stateSize] = [permute.finalState[0], permute.finalState[1] +  point.x.limbs[2], permute.finalState[2] +  point.y.limbs[0],  permute.finalState[3] +point.y.limbs[1], permute.finalState[4] + point.y.limbs[2]];
    component permute_y = permute(1);
    permute_y.state <-- int_state_y;


    up_transcript.state <== permute_y.finalState[capacity];
    up_transcript.nRounds <== transcript.nRounds + 1;
   
}

template AppendPoints(nPoints){
    input G1Affine() points[nPoints];
    input Transcript() transcript;
    output Transcript() up_transcript;

    Transcript() int_transcript[nPoints + 1];

    int_transcript[0] <== transcript;  

   for (var i = 0; i < nPoints; i++)
   {
        int_transcript[i + 1] <== AppendPoint()(points[i], int_transcript[i]);
   }
   
    up_transcript <== int_transcript[nPoints];
}

template ChallengeScalar() {
    input Transcript() transcript;
    output Transcript() up_transcript;
    signal output challenge;
 
    var stateSize = getStateSize();
    var capacity = getCapacity();

    var int_state[stateSize] = [0, transcript.state, transcript.nRounds, 0, 0];
   
    component permute = permute(1);
    permute.state <-- int_state;
   
    up_transcript.state <== permute.finalState[capacity];
    up_transcript.nRounds <== transcript.nRounds + 1;
    
    challenge <== up_transcript.state;

}

template ChallengeVector(len) {
    input Transcript() transcript;
    output Transcript() up_transcript;
    signal output challenges[len];
    
    Transcript() int_transcripts[len];

    (int_transcripts[0],  challenges[0]) <== ChallengeScalar()(transcript);
    for (var i = 1; i < len; i++)
    {
       (int_transcripts[i], challenges[i]) <== ChallengeScalar()(int_transcripts[i - 1]);
    }

    up_transcript <== int_transcripts[len - 1];
}

template ChallengeScalarPowers(len){
    input Transcript() transcript;
    output Transcript() up_transcript;
    output signal challenges[len];

    (up_transcript, challenges[1]) <== ChallengeScalar()(transcript);
    challenges[0] <== 1;

    for (var i = 2; i < len; i++){
        challenges[i] <== challenges[i - 1] * challenges[1];
    }
}

//  component main = ChallengeScalarPowers(3);
//  component main = AppendScalar();
//  component main = AppendBytes(6);
//  component main = FiatShamirPreamble(1, 2);