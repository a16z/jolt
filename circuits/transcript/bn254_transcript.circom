
// Poseidon hash is over Bn254 Basefield
pragma circom 2.2.1;
include "./poseidon.circom";
include "./../fields/non_native/utils.circom";
include "./../fields/non_native/non_native_over_bn_base.circom";
include "./../groups/utils.circom";

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

    component permute = permute(2);
    permute.state <-- int_state;

    new_transcript.state <== permute.finalState[capacity];
    log("new state = ", new_transcript.state);
    new_transcript.nRounds <== 0;
}

template AppendScalar() {
    input Fq() scalar;
    input Transcript() transcript;
    output Transcript() up_transcript;

    var stateSize = getStateSize();
    var capacity = getCapacity();
    signal element <== scalar.limbs[0] + (1 << 125) * scalar.limbs[1] + (1 << 250) * scalar.limbs[2];
    var int_state[stateSize] = [0, transcript.state, transcript.nRounds, element, 0];
    
 
    component permute = permute(2);
    permute.state <-- int_state;

    up_transcript.state <== permute.finalState[capacity];
    up_transcript.nRounds <== transcript.nRounds + 1;
}

template AppendScalars(nScalars) {
    input Fq() scalars[nScalars];
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

template AppendPoint(){
    input G1Affine() point;
    input Transcript() transcript;
    output Transcript() up_transcript;

    var stateSize = getStateSize();
    var capacity = getCapacity();

    var int_state[stateSize] = [0, transcript.state, transcript.nRounds, point.x, point.y];

    component permute = permute(2);
    permute.state <-- int_state; 
   
    up_transcript.state <== permute.finalState[capacity];
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
    Fq() output challenge;
 
    var stateSize = getStateSize();
    var capacity = getCapacity();

    var int_state[stateSize] = [0, transcript.state, transcript.nRounds, 0, 0];
   
    component permute = permute(2);
    permute.state <-- int_state;
   
    up_transcript.state <== permute.finalState[capacity];
    up_transcript.nRounds <== transcript.nRounds + 1;
    
    challenge <== NonNativeModuloFp()(up_transcript.state);

}

template ChallengeVector(len) {
    input Transcript() transcript;
    output Transcript() up_transcript;
    Fq() output challenges[len];
    
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
    output Fq() challenges[len];
    
    Fq() one; 
    one.limbs  <== [1, 0, 0];

    (up_transcript, challenges[1]) <== ChallengeScalar()(transcript);
    challenges[0] <== one;

    for (var i = 2; i < len; i++){
        challenges[i] <== NonNativeMul()(challenges[i - 1], challenges[1]);
    }
}

// component main = ChallengeScalarPowers(3);
// component main = AppendScalar();
// component main = FiatShamirPreamble(1, 2);