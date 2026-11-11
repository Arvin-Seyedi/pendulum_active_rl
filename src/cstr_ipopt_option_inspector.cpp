// Read-only inspection of the constructed CasADi 3.7.2 IPOPT application's options.
// Uses exact-version internal layout headers and public IPOPT option getters.
// Never calls OptimizeTNLP, Initialize, or any solver/integrator evaluation.
#include <casadi/casadi.hpp>
#include <casadi/interfaces/ipopt/ipopt_interface.hpp>
#include <coin-or/IpIpoptApplication.hpp>
#include <coin-or/IpOptionsList.hpp>
#include <cstdint>
#include <cstdio>
#include <exception>

extern "C" int cstr_ipopt_options(std::uintptr_t address, double* values,
                                  char* error, std::size_t length) {
  auto* function = reinterpret_cast<casadi::Function*>(address);
  int slot = -1;
  try {
    if (!function || !values) throw std::runtime_error("Null inspector argument");
    slot = static_cast<int>(function->checkout());
    auto* memory = static_cast<casadi::IpoptMemory*>(function->memory(slot));
    if (!memory || !memory->app) throw std::runtime_error("No IPOPT application");
    auto* app = static_cast<Ipopt::SmartPtr<Ipopt::IpoptApplication>*>(memory->app);
    if (!Ipopt::IsValid(*app)) throw std::runtime_error("Invalid IPOPT application");
    const char* names[] = {"bound_relax_factor", "tol", "constr_viol_tol",
                          "dual_inf_tol", "compl_inf_tol", "acceptable_tol"};
    for (int i=0; i<6; ++i) {
      (*app)->Options()->GetNumericValue(names[i], values[i], "");
      // GetNumericValue's bool indicates explicit-setting versus default;
      // defaults still populate the value and are valid observations.
    }
    std::string honor;
    (*app)->Options()->GetStringValue("honor_original_bounds", honor, "");
    values[6] = honor == "yes" ? 1.0 : (honor == "no" ? 0.0 : -1.0);
    function->release(slot); slot = -1;
    return 0;
  } catch (const std::exception& ex) {
    if (slot >= 0) function->release(slot);
    std::snprintf(error, length, "%s", ex.what());
    return 1;
  }
}
