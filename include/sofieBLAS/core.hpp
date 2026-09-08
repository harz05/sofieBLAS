#pragma once

namespace traits {
template <typename TTag> class sofieBLAS;
}

template <typename TTag>
using sofieBLAS = typename traits::sofieBLAS<TTag>::Impl;

enum class Epilogue { Default, Bias, ReluBias, GeluBias };
