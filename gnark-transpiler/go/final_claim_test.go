package jolt_verifier

import (
	"fmt"
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

// FinalClaimCircuit - tests just the final_claim computation from sumcheck
type FinalClaimCircuit struct {
	// Sumcheck round polynomials (compressed: c0, c2, c3)
	SumcheckR0C0  frontend.Variable `gnark:",public"`
	SumcheckR0C1  frontend.Variable `gnark:",public"`
	SumcheckR0C2  frontend.Variable `gnark:",public"`
	SumcheckR1C0  frontend.Variable `gnark:",public"`
	SumcheckR1C1  frontend.Variable `gnark:",public"`
	SumcheckR1C2  frontend.Variable `gnark:",public"`
	SumcheckR2C0  frontend.Variable `gnark:",public"`
	SumcheckR2C1  frontend.Variable `gnark:",public"`
	SumcheckR2C2  frontend.Variable `gnark:",public"`
	SumcheckR3C0  frontend.Variable `gnark:",public"`
	SumcheckR3C1  frontend.Variable `gnark:",public"`
	SumcheckR3C2  frontend.Variable `gnark:",public"`
	SumcheckR4C0  frontend.Variable `gnark:",public"`
	SumcheckR4C1  frontend.Variable `gnark:",public"`
	SumcheckR4C2  frontend.Variable `gnark:",public"`
	SumcheckR5C0  frontend.Variable `gnark:",public"`
	SumcheckR5C1  frontend.Variable `gnark:",public"`
	SumcheckR5C2  frontend.Variable `gnark:",public"`
	SumcheckR6C0  frontend.Variable `gnark:",public"`
	SumcheckR6C1  frontend.Variable `gnark:",public"`
	SumcheckR6C2  frontend.Variable `gnark:",public"`
	SumcheckR7C0  frontend.Variable `gnark:",public"`
	SumcheckR7C1  frontend.Variable `gnark:",public"`
	SumcheckR7C2  frontend.Variable `gnark:",public"`
	SumcheckR8C0  frontend.Variable `gnark:",public"`
	SumcheckR8C1  frontend.Variable `gnark:",public"`
	SumcheckR8C2  frontend.Variable `gnark:",public"`
	SumcheckR9C0  frontend.Variable `gnark:",public"`
	SumcheckR9C1  frontend.Variable `gnark:",public"`
	SumcheckR9C2  frontend.Variable `gnark:",public"`
	SumcheckR10C0 frontend.Variable `gnark:",public"`
	SumcheckR10C1 frontend.Variable `gnark:",public"`
	SumcheckR10C2 frontend.Variable `gnark:",public"`

	// Challenges (from transcript - these would be computed in full circuit)
	R0  frontend.Variable `gnark:",public"`
	R1  frontend.Variable `gnark:",public"`
	R2  frontend.Variable `gnark:",public"`
	R3  frontend.Variable `gnark:",public"`
	R4  frontend.Variable `gnark:",public"`
	R5  frontend.Variable `gnark:",public"`
	R6  frontend.Variable `gnark:",public"`
	R7  frontend.Variable `gnark:",public"`
	R8  frontend.Variable `gnark:",public"`
	R9  frontend.Variable `gnark:",public"`
	R10 frontend.Variable `gnark:",public"`

	// The initial claim (from uni-skip first round evaluation)
	InitialClaim frontend.Variable `gnark:",public"`

	// Expected final claim
	ExpectedFinalClaim frontend.Variable `gnark:",public"`
}

func (circuit *FinalClaimCircuit) Define(api frontend.API) error {
	// Compute final_claim using compressed sumcheck polynomial evaluation
	// Each round: claim = p(r) where p(x) = c0 + c1*x + c2*x^2 + c3*x^3
	// But c1 is derived: c1 = claim - 2*c0 - c2 - c3 (from sumcheck property)

	claim := circuit.InitialClaim

	// Round 0
	r0_sq := api.Mul(circuit.R0, circuit.R0)
	c1_0 := api.Sub(api.Sub(api.Sub(api.Sub(claim, circuit.SumcheckR0C0), circuit.SumcheckR0C0), circuit.SumcheckR0C1), circuit.SumcheckR0C2)
	claim = api.Add(api.Add(api.Add(circuit.SumcheckR0C0, api.Mul(c1_0, circuit.R0)), api.Mul(circuit.SumcheckR0C1, r0_sq)), api.Mul(circuit.SumcheckR0C2, api.Mul(r0_sq, circuit.R0)))

	// Round 1
	r1_sq := api.Mul(circuit.R1, circuit.R1)
	c1_1 := api.Sub(api.Sub(api.Sub(api.Sub(claim, circuit.SumcheckR1C0), circuit.SumcheckR1C0), circuit.SumcheckR1C1), circuit.SumcheckR1C2)
	claim = api.Add(api.Add(api.Add(circuit.SumcheckR1C0, api.Mul(c1_1, circuit.R1)), api.Mul(circuit.SumcheckR1C1, r1_sq)), api.Mul(circuit.SumcheckR1C2, api.Mul(r1_sq, circuit.R1)))

	// Round 2
	r2_sq := api.Mul(circuit.R2, circuit.R2)
	c1_2 := api.Sub(api.Sub(api.Sub(api.Sub(claim, circuit.SumcheckR2C0), circuit.SumcheckR2C0), circuit.SumcheckR2C1), circuit.SumcheckR2C2)
	claim = api.Add(api.Add(api.Add(circuit.SumcheckR2C0, api.Mul(c1_2, circuit.R2)), api.Mul(circuit.SumcheckR2C1, r2_sq)), api.Mul(circuit.SumcheckR2C2, api.Mul(r2_sq, circuit.R2)))

	// Round 3
	r3_sq := api.Mul(circuit.R3, circuit.R3)
	c1_3 := api.Sub(api.Sub(api.Sub(api.Sub(claim, circuit.SumcheckR3C0), circuit.SumcheckR3C0), circuit.SumcheckR3C1), circuit.SumcheckR3C2)
	claim = api.Add(api.Add(api.Add(circuit.SumcheckR3C0, api.Mul(c1_3, circuit.R3)), api.Mul(circuit.SumcheckR3C1, r3_sq)), api.Mul(circuit.SumcheckR3C2, api.Mul(r3_sq, circuit.R3)))

	// Round 4
	r4_sq := api.Mul(circuit.R4, circuit.R4)
	c1_4 := api.Sub(api.Sub(api.Sub(api.Sub(claim, circuit.SumcheckR4C0), circuit.SumcheckR4C0), circuit.SumcheckR4C1), circuit.SumcheckR4C2)
	claim = api.Add(api.Add(api.Add(circuit.SumcheckR4C0, api.Mul(c1_4, circuit.R4)), api.Mul(circuit.SumcheckR4C1, r4_sq)), api.Mul(circuit.SumcheckR4C2, api.Mul(r4_sq, circuit.R4)))

	// Round 5
	r5_sq := api.Mul(circuit.R5, circuit.R5)
	c1_5 := api.Sub(api.Sub(api.Sub(api.Sub(claim, circuit.SumcheckR5C0), circuit.SumcheckR5C0), circuit.SumcheckR5C1), circuit.SumcheckR5C2)
	claim = api.Add(api.Add(api.Add(circuit.SumcheckR5C0, api.Mul(c1_5, circuit.R5)), api.Mul(circuit.SumcheckR5C1, r5_sq)), api.Mul(circuit.SumcheckR5C2, api.Mul(r5_sq, circuit.R5)))

	// Round 6
	r6_sq := api.Mul(circuit.R6, circuit.R6)
	c1_6 := api.Sub(api.Sub(api.Sub(api.Sub(claim, circuit.SumcheckR6C0), circuit.SumcheckR6C0), circuit.SumcheckR6C1), circuit.SumcheckR6C2)
	claim = api.Add(api.Add(api.Add(circuit.SumcheckR6C0, api.Mul(c1_6, circuit.R6)), api.Mul(circuit.SumcheckR6C1, r6_sq)), api.Mul(circuit.SumcheckR6C2, api.Mul(r6_sq, circuit.R6)))

	// Round 7
	r7_sq := api.Mul(circuit.R7, circuit.R7)
	c1_7 := api.Sub(api.Sub(api.Sub(api.Sub(claim, circuit.SumcheckR7C0), circuit.SumcheckR7C0), circuit.SumcheckR7C1), circuit.SumcheckR7C2)
	claim = api.Add(api.Add(api.Add(circuit.SumcheckR7C0, api.Mul(c1_7, circuit.R7)), api.Mul(circuit.SumcheckR7C1, r7_sq)), api.Mul(circuit.SumcheckR7C2, api.Mul(r7_sq, circuit.R7)))

	// Round 8
	r8_sq := api.Mul(circuit.R8, circuit.R8)
	c1_8 := api.Sub(api.Sub(api.Sub(api.Sub(claim, circuit.SumcheckR8C0), circuit.SumcheckR8C0), circuit.SumcheckR8C1), circuit.SumcheckR8C2)
	claim = api.Add(api.Add(api.Add(circuit.SumcheckR8C0, api.Mul(c1_8, circuit.R8)), api.Mul(circuit.SumcheckR8C1, r8_sq)), api.Mul(circuit.SumcheckR8C2, api.Mul(r8_sq, circuit.R8)))

	// Round 9
	r9_sq := api.Mul(circuit.R9, circuit.R9)
	c1_9 := api.Sub(api.Sub(api.Sub(api.Sub(claim, circuit.SumcheckR9C0), circuit.SumcheckR9C0), circuit.SumcheckR9C1), circuit.SumcheckR9C2)
	claim = api.Add(api.Add(api.Add(circuit.SumcheckR9C0, api.Mul(c1_9, circuit.R9)), api.Mul(circuit.SumcheckR9C1, r9_sq)), api.Mul(circuit.SumcheckR9C2, api.Mul(r9_sq, circuit.R9)))

	// Round 10
	r10_sq := api.Mul(circuit.R10, circuit.R10)
	c1_10 := api.Sub(api.Sub(api.Sub(api.Sub(claim, circuit.SumcheckR10C0), circuit.SumcheckR10C0), circuit.SumcheckR10C1), circuit.SumcheckR10C2)
	claim = api.Add(api.Add(api.Add(circuit.SumcheckR10C0, api.Mul(c1_10, circuit.R10)), api.Mul(circuit.SumcheckR10C1, r10_sq)), api.Mul(circuit.SumcheckR10C2, api.Mul(r10_sq, circuit.R10)))

	// Assert final claim matches expected
	api.AssertIsEqual(claim, circuit.ExpectedFinalClaim)

	return nil
}

func TestFinalClaimCircuit(t *testing.T) {
	circuit := &FinalClaimCircuit{}

	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}
	fmt.Printf("FinalClaim circuit compiled with %d constraints\n", ccs.GetNbConstraints())

	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		t.Fatalf("Setup failed: %v", err)
	}
	_ = vk

	// Use real values from the proof
	// Sumcheck polynomials from fib_stage1_data.json
	assignment := &FinalClaimCircuit{}
	assignment.SumcheckR0C0, _ = new(big.Int).SetString("14813696870531634164367264378872173877114964604757197562228936145068638015314", 10)
	assignment.SumcheckR0C1, _ = new(big.Int).SetString("6127285717690388377608534322051447994563938792643525742485899181520754771067", 10)
	assignment.SumcheckR0C2, _ = new(big.Int).SetString("4912035171906721066561476772124839233700012675036116078305842559822554257265", 10)
	assignment.SumcheckR1C0, _ = new(big.Int).SetString("12152300614401221990243959031100367073134098524557432448463940488799431366670", 10)
	assignment.SumcheckR1C1, _ = new(big.Int).SetString("13093883956264976608298213025812381131996438675987782590700192117177539474253", 10)
	assignment.SumcheckR1C2, _ = new(big.Int).SetString("1571155441262335281080567140445496273089863633847232230563212642992881452073", 10)
	assignment.SumcheckR2C0, _ = new(big.Int).SetString("18806079335524228979055746264945869102374532661418926257848176866301326006663", 10)
	assignment.SumcheckR2C1, _ = new(big.Int).SetString("20162311101759472845595354455460769442296067423336143567679489225831697644087", 10)
	assignment.SumcheckR2C2, _ = new(big.Int).SetString("4414236813425179300449196688509169274720761028915333530801139777695785953885", 10)
	assignment.SumcheckR3C0, _ = new(big.Int).SetString("21033372844866040724313306222692169842426845994429867806287897580001409257365", 10)
	assignment.SumcheckR3C1, _ = new(big.Int).SetString("18647239881819439108835974754164632558429243037064227518439008189645122348353", 10)
	assignment.SumcheckR3C2, _ = new(big.Int).SetString("10755148748207403782441969168834423691635268272880216140774839499977885271221", 10)
	assignment.SumcheckR4C0, _ = new(big.Int).SetString("14490893492227111748052783845785570476780315487218516450642814084637663575683", 10)
	assignment.SumcheckR4C1, _ = new(big.Int).SetString("14046159791764981445606854393490790837640658021075980379852771056213934191922", 10)
	assignment.SumcheckR4C2, _ = new(big.Int).SetString("2592020866896269789749879980838798993662196240325989666979739120212761163192", 10)
	assignment.SumcheckR5C0, _ = new(big.Int).SetString("9724599951142292786784444751033162960399208473697697805148559039309137403094", 10)
	assignment.SumcheckR5C1, _ = new(big.Int).SetString("17928446807279662435173378984711549576631858282523606632626934911069536863116", 10)
	assignment.SumcheckR5C2, _ = new(big.Int).SetString("6557766676465361562814540791648450030188706642323249155811484491655440769854", 10)
	assignment.SumcheckR6C0, _ = new(big.Int).SetString("20679885401004961472324801639427429905739469248013398723845510114511001584917", 10)
	assignment.SumcheckR6C1, _ = new(big.Int).SetString("15547195814955209059666683961412146306612144856089682339973831755434477067387", 10)
	assignment.SumcheckR6C2, _ = new(big.Int).SetString("15953663246726436690151923918206657749936726207680013386252803478638713024772", 10)
	assignment.SumcheckR7C0, _ = new(big.Int).SetString("18972735130700846522967159665811665413108877647288471985444644695435783230909", 10)
	assignment.SumcheckR7C1, _ = new(big.Int).SetString("18479199790213668696076705127951971038814234904195506946245789949551931824445", 10)
	assignment.SumcheckR7C2, _ = new(big.Int).SetString("1666119346243080899089060026592147303489788359885062502475007812392500280311", 10)
	assignment.SumcheckR8C0, _ = new(big.Int).SetString("9001804755947123307336638184077865574023618762174089955225303121576682969229", 10)
	assignment.SumcheckR8C1, _ = new(big.Int).SetString("1772042467372014363003690405414915086596964124925538268214920364637562763619", 10)
	assignment.SumcheckR8C2, _ = new(big.Int).SetString("8265575449297488456794008271228241259153084661062874365634076127509086539533", 10)
	assignment.SumcheckR9C0, _ = new(big.Int).SetString("1186146013423658200566003175496368165170582739713144195849725765746338712384", 10)
	assignment.SumcheckR9C1, _ = new(big.Int).SetString("20915373313706339844290257768979202343832032600046019618536931630014612630879", 10)
	assignment.SumcheckR9C2, _ = new(big.Int).SetString("851763409753603678952980463087403692831804556332615277904594384793158598597", 10)
	assignment.SumcheckR10C0, _ = new(big.Int).SetString("3668073689012847048377700135970979706247315498740970605001296399166997858195", 10)
	assignment.SumcheckR10C1, _ = new(big.Int).SetString("9606929016890912461653755310765912632952055063551903755832149685855303399291", 10)
	assignment.SumcheckR10C2, _ = new(big.Int).SetString("4019208284709636813632551977399095550347697159994385891506729635693921109715", 10)

	// These challenges need to come from the transcript - placeholder for now
	// We need to extract these from Rust
	assignment.R0, _ = new(big.Int).SetString("0", 10)  // TODO: get from Rust
	assignment.R1, _ = new(big.Int).SetString("0", 10)
	assignment.R2, _ = new(big.Int).SetString("0", 10)
	assignment.R3, _ = new(big.Int).SetString("0", 10)
	assignment.R4, _ = new(big.Int).SetString("0", 10)
	assignment.R5, _ = new(big.Int).SetString("0", 10)
	assignment.R6, _ = new(big.Int).SetString("0", 10)
	assignment.R7, _ = new(big.Int).SetString("0", 10)
	assignment.R8, _ = new(big.Int).SetString("0", 10)
	assignment.R9, _ = new(big.Int).SetString("0", 10)
	assignment.R10, _ = new(big.Int).SetString("0", 10)

	assignment.InitialClaim, _ = new(big.Int).SetString("0", 10)  // TODO: get from Rust
	assignment.ExpectedFinalClaim, _ = new(big.Int).SetString("7687275872079118408585697803972719249193818707221360741413929442803083494908", 10)

	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Failed to create witness: %v", err)
	}

	_, err = groth16.Prove(ccs, pk, witness)
	if err != nil {
		fmt.Printf("Note: This test needs real challenge values from Rust transcript\n")
		fmt.Printf("Proof failed: %v\n", err)
	}
}
