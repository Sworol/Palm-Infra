#include "engine/sampler.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void test_greedy_and_filters() {
    {
        SamplingParams p;
        p.temperature = 0.0f;
        Sampler sampler(p);
        expect(sampler.uses_plain_argmax(),
               "temperature=0 exposes the plain-argmax fast path");
        float logits[] = {0.5f, 3.0f, 2.0f};
        expect(sampler.sample(logits, 3) == 1, "temperature=0 is greedy");
    }
    {
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 1;
        Sampler sampler(p);
        expect(sampler.uses_plain_argmax(),
               "top_k=1 exposes the plain-argmax fast path");
        float logits[] = {0.5f, 3.0f, 2.0f};
        expect(sampler.sample(logits, 3) == 1, "top_k=1 is greedy");
    }
    {
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 0;
        p.top_p = 0.0f;
        p.min_p = 0.5f;
        Sampler sampler(p);
        float logits[] = {4.0f, 0.0f, -1.0f};
        expect(sampler.sample(logits, 3) == 0,
               "min_p removes tokens far below the best probability");
    }
    {
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 0;
        p.top_p = 0.5f;
        Sampler sampler(p);
        float logits[] = {4.0f, 0.0f, -1.0f};
        expect(sampler.sample(logits, 3) == 0,
               "top_p retains the minimum prefix reaching the cutoff");
    }
}

void test_penalties_and_logit_restore() {
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.presence_penalty = 1.0f;
        Sampler sampler(p);
        expect(!sampler.uses_plain_argmax(),
               "penalties disable the plain-argmax fast path");
        sampler.accept(0);
        float logits[] = {3.0f, 2.5f};
        expect(sampler.sample(logits, 2) == 1,
               "presence penalty changes greedy selection");
        expect(logits[0] == 3.0f && logits[1] == 2.5f,
               "presence penalty restores caller logits");
    }
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.frequency_penalty = 0.5f;
        Sampler sampler(p);
        sampler.accept(0);
        sampler.accept(0);
        float logits[] = {3.0f, 2.5f};
        expect(sampler.sample(logits, 2) == 1,
               "frequency penalty scales with token count");
    }
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.repeat_penalty = 2.0f;
        p.repeat_last_n = 8;
        Sampler sampler(p);
        sampler.accept(0);
        float logits[] = {3.0f, 2.5f};
        expect(sampler.sample(logits, 2) == 1,
               "repeat penalty applies inside the recent window");
    }
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.presence_penalty = 1.0f;
        p.repeat_last_n = 1;
        Sampler sampler(p);
        sampler.accept(0);
        sampler.accept(1);
        float logits[] = {3.0f, 1.0f, 2.5f};
        expect(sampler.sample(logits, 3) == 0,
               "presence penalty ignores tokens outside repeat_last_n");
    }
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.frequency_penalty = 1.0f;
        p.repeat_last_n = 2;
        Sampler sampler(p);
        sampler.accept(0);
        sampler.accept(0);
        sampler.accept(1);
        float logits[] = {4.0f, 1.0f, 2.5f};
        expect(sampler.sample(logits, 3) == 0,
               "frequency counts only tokens inside repeat_last_n");
    }
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.presence_penalty = 2.0f;
        p.frequency_penalty = 2.0f;
        p.repeat_penalty = 2.0f;
        p.repeat_last_n = 0;
        Sampler sampler(p);
        sampler.accept(0);
        float logits[] = {3.0f, 2.5f};
        expect(sampler.sample(logits, 2) == 0,
               "repeat_last_n=0 disables all repetition penalties");
    }
}

void test_seed_and_reset() {
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k = 3;
    p.top_p = 1.0f;
    p.seed = 1234;
    Sampler a(p), b(p);
    for (int i = 0; i < 32; ++i) {
        float logits_a[] = {1.0f, 0.9f, 0.8f};
        float logits_b[] = {1.0f, 0.9f, 0.8f};
        const int ta = a.sample(logits_a, 3);
        const int tb = b.sample(logits_b, 3);
        expect(ta == tb, "same seed produces the same sample stream");
        a.accept(ta);
        b.accept(tb);
    }
    a.reset();
    b.reset();
    expect(a.history_size() == 0 && b.history_size() == 0,
           "reset clears sampling history");
    float logits_a[] = {1.0f, 0.9f, 0.8f};
    float logits_b[] = {1.0f, 0.9f, 0.8f};
    expect(a.sample(logits_a, 3) == b.sample(logits_b, 3),
           "reset restores the configured seed");
}

void test_probability_and_rejection_sampling() {
    {
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 0;
        p.top_p = 1.0f;
        Sampler sampler(p);
        const float logits[] = {0.0f, std::log(2.0f), std::log(3.0f)};
        std::vector<float> probabilities;
        sampler.probabilities(logits, 3, {}, probabilities);
        expect(probabilities.size() == 3,
               "probability builder returns the complete vocabulary");
        expect(std::fabs(probabilities[0] - 1.0f / 6.0f) < 1e-5f &&
                   std::fabs(probabilities[1] - 2.0f / 6.0f) < 1e-5f &&
                   std::fabs(probabilities[2] - 3.0f / 6.0f) < 1e-5f,
               "probability builder matches temperature softmax");
    }
    {
        SamplingParams p;
        p.temperature = 0.0f;
        p.presence_penalty = 2.0f;
        Sampler sampler(p);
        const float logits[] = {3.0f, 2.0f};
        std::vector<float> probabilities;
        sampler.probabilities(logits, 2, {0}, probabilities);
        expect(probabilities[0] == 0.0f && probabilities[1] == 1.0f,
               "uncommitted proposal history participates in penalties");
        expect(sampler.history_size() == 0,
               "probability construction does not commit proposal history");
    }
    {
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 0;
        p.top_p = 1.0f;
        p.seed = 1234;
        Sampler sampler(p);
        const std::vector<float> target = {0.1f, 0.3f, 0.6f};
        const std::vector<float> proposal = {0.5f, 0.4f, 0.1f};
        int counts[3] = {0, 0, 0};
        constexpr int trials = 50000;
        for (int i = 0; i < trials; ++i) {
            const int proposed = sampler.sample_probabilities(proposal);
            bool accepted = false;
            const int sampled = sampler.speculative_sample(
                target, proposal, proposed, &accepted);
            (void)accepted;
            ++counts[sampled];
        }
        for (int token = 0; token < 3; ++token) {
            const float observed =
                static_cast<float>(counts[token]) / trials;
            expect(std::fabs(observed - target[token]) < 0.01f,
                   "p/q rejection sampling reproduces the target distribution");
        }
    }
}

void test_validation() {
    SamplingParams p;
    std::string error;
    p.temperature = 2.1f;
    expect(!validate_sampling_params(p, &error) && !error.empty(),
           "temperature range is validated");
    p = SamplingParams{};
    p.repeat_last_n = -2;
    expect(!validate_sampling_params(p, &error),
           "repeat_last_n range is validated");
    p = SamplingParams{};
    p.frequency_penalty = -2.0f;
    p.presence_penalty = 2.0f;
    expect(validate_sampling_params(p, &error),
           "OpenAI penalty boundary values are accepted");
}

void test_greedy_ties_across_vector_boundaries() {
    // Exercise scalar tails and ties across NEON lanes on the ARM CI jobs.
    for (int size : {1, 3, 4, 5, 8, 9, 17}) {
        for (int first = 0; first < size; ++first) {
            std::vector<float> logits(size, -5.0f);
            logits[first] = 2.0f;
            logits.back() = 2.0f;
            for (bool use_top_k : {false, true}) {
                SamplingParams p;
                p.temperature = use_top_k ? 1.0f : 0.0f;
                p.top_k = use_top_k ? 1 : 0;
                Sampler sampler(p);
                expect(sampler.sample(logits.data(), size) == first,
                       "greedy ties select the lowest token ID across vector boundaries");
            }
        }
    }
}

void test_filter_distributions_and_numerical_stability() {
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k = 0;
    p.top_p = 1.0f;
    Sampler sampler(p);
    const float logits[] = {0.0f, -1.0f, -2.0f, -10000.0f};
    std::vector<float> reference, shifted;
    sampler.probabilities(logits, 4, {}, reference);
    for (float offset : {-10000.0f, 10000.0f}) {
        float values[4];
        for (int i = 0; i < 4; ++i)
            values[i] = logits[i] + offset;
        sampler.probabilities(values, 4, {}, shifted);
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            expect(std::isfinite(shifted[i]) && shifted[i] >= 0.0f &&
                       std::fabs(shifted[i] - reference[i]) < 1e-6f,
                   "softmax is finite and invariant under a large common logit offset");
            sum += shifted[i];
        }
        expect(std::fabs(sum - 1.0f) < 1e-6f,
               "extreme finite logits still produce normalized probabilities");
    }

    const float tied[] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k : {2, 10}) {
        p.top_k = k;
        sampler.configure(p);
        sampler.probabilities(tied, 4, {}, shifted);
        const int retained = k == 2 ? 2 : 4;
        for (int i = 0; i < 4; ++i)
            expect(shifted[i] == (i < retained ? 1.0f / retained : 0.0f),
                   "top-k resolves ties by token ID and tolerates k above vocabulary size");
    }
    p.top_k = 0;
    p.top_p = 0.5f;
    sampler.configure(p);
    sampler.probabilities(tied, 4, {}, shifted);
    expect(shifted == std::vector<float>({0.5f, 0.5f, 0.0f, 0.0f}),
           "top-p includes exactly the prefix reaching an exact cutoff");
}

void test_negative_repeat_penalty_and_reset() {
    SamplingParams p;
    p.temperature = 0.0f;
    p.repeat_penalty = 2.0f;
    p.repeat_last_n = -1;
    Sampler sampler(p);
    sampler.accept(0);
    sampler.accept(std::vector<int>(128, 2));
    float logits[] = {-1.0f, -1.5f, -10.0f};
    expect(sampler.sample(logits, 3) == 1,
           "full-context repeat penalty multiplies negative logits even beyond 64 tokens");
    expect(logits[0] == -1.0f && logits[1] == -1.5f && logits[2] == -10.0f,
           "negative repeat penalties restore every caller logit");
    sampler.reset();
    expect(sampler.sample(logits, 3) == 0,
           "reset removes repetition penalties from a previous conversation");
}

void test_reconfiguration_preserves_state_on_failure() {
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k = 0;
    p.top_p = 1.0f;
    p.presence_penalty = 0.5f;
    Sampler sampler(p), reference(p);
    float logits[] = {1.0f, 0.9f, 0.8f};
    for (int i = 0; i < 7; ++i) {
        sampler.accept(sampler.sample(logits, 3));
        reference.accept(reference.sample(logits, 3));
    }
    // Invalid runtime configuration must not reset the RNG or token history.
    for (auto member : {&SamplingParams::temperature, &SamplingParams::top_p,
                        &SamplingParams::min_p, &SamplingParams::repeat_penalty,
                        &SamplingParams::presence_penalty,
                        &SamplingParams::frequency_penalty}) {
        for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
            SamplingParams bad = p;
            bad.*member = invalid;
            bad.seed = 9876;
            std::string error;
            expect(!sampler.configure(bad, &error) && !error.empty(),
                   "non-finite sampling parameters are rejected with a diagnostic");
            expect(sampler.history_size() == reference.history_size(),
                   "failed configuration preserves token history");
            for (int i = 0; i < 16; ++i)
                expect(sampler.sample(logits, 3) == reference.sample(logits, 3),
                       "failed configuration preserves sampling parameters and RNG state");
        }
    }
    sampler.reset();
    Sampler fresh(p);
    for (int i = 0; i < 32; ++i)
        expect(sampler.sample(logits, 3) == fresh.sample(logits, 3),
               "reset reproduces a fresh sampler's stream");
}

} // namespace

int main() {
    test_greedy_and_filters();
    test_penalties_and_logit_restore();
    test_seed_and_reset();
    test_probability_and_rejection_sampling();
    test_validation();
    test_greedy_ties_across_vector_boundaries();
    test_filter_distributions_and_numerical_stability();
    test_negative_repeat_penalty_and_reset();
    test_reconfiguration_preserves_state_on_failure();
    if (failures != 0)
        return 1;
    std::puts("All sampler tests passed.");
    return 0;
}
