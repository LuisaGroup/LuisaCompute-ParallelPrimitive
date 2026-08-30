// Tests for DeviceTransform.
// Covers fill/generate, unary and tuple transforms, multiple outputs,
// conditional writes, exact in-place execution, and stable-address aliases.

#include <lcpp/parallel_primitive.h>

#include <boost/ut.hpp>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

#include <tuple>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::parallel_primitive;
using boost::ut::expect;
using boost::ut::operator""_test;

int main(int argc, char* argv[])
{
    luisa::string_view context_directory;
    luisa::string_view backend_name = "cuda";
    for(int i = 1; i < argc; ++i)
    {
        luisa::string_view argument(argv[i]);
        if(argument.starts_with("--ctx="))
        {
            context_directory = argument.substr(6);
        }
        else if(argument.starts_with("--backend="))
        {
            backend_name = argument.substr(10);
        }
    }

    Context context{context_directory.empty() ? argv[0] : context_directory};
    Device  device = context.create_device(backend_name);
    Stream  stream = device.create_stream();

    DeviceTransform<> transform;
    transform.create(device);

    "device_transform_fill_and_generate"_test = [&]
    {
        constexpr size_t item_count = 8u;
        expect(DeviceTransform<>::GetTempStorageBytes<int>(item_count) == 0u);
        auto filled             = device.create_buffer<int>(item_count);
        auto converted_fill     = device.create_buffer<float>(item_count);
        auto generated          = device.create_buffer<int>(item_count);
        auto converted_generate = device.create_buffer<float>(item_count);

        auto        generator = []() noexcept { return cast<int>(dispatch_id().x) * 3; };
        CommandList commands;
        transform.Fill(commands, BufferView<uint>{}, filled.view(), item_count, 7);
        transform.Fill(commands, BufferView<uint>{}, converted_fill.view(), item_count, 5);
        transform.Generate(commands, BufferView<uint>{}, generated.view(), item_count, generator);
        transform.Generate(commands, BufferView<uint>{}, converted_generate.view(), item_count, generator);
        transform.Fill(commands, BufferView<uint>{}, BufferView<int>{}, 0u, 11);

        luisa::vector<int>   filled_result(item_count);
        luisa::vector<float> converted_fill_result(item_count);
        luisa::vector<int>   generated_result(item_count);
        luisa::vector<float> converted_generate_result(item_count);
        stream << commands.commit() << filled.copy_to(luisa::span{filled_result})
               << converted_fill.copy_to(luisa::span{converted_fill_result})
               << generated.copy_to(luisa::span{generated_result})
               << converted_generate.copy_to(luisa::span{converted_generate_result}) << synchronize();
        for(size_t index = 0u; index < item_count; ++index)
        {
            expect(filled_result[index] == 7);
            expect(converted_fill_result[index] == 5.0f);
            expect(generated_result[index] == static_cast<int>(index * 3u));
            expect(converted_generate_result[index] == static_cast<float>(index * 3u));
        }
    };

    "device_transform_supports_unary_and_tuple_dataflows"_test = [&]
    {
        const luisa::vector<int> unary_values{1, 2, 3, 4};
        const luisa::vector<int> input_a_values{1, 2, 3, 4};
        const luisa::vector<int> input_b_values{10, 20, 30, 40};
        auto                     unary      = device.create_buffer<int>(unary_values.size());
        auto                     input_a    = device.create_buffer<int>(input_a_values.size());
        auto                     input_b    = device.create_buffer<int>(input_b_values.size());
        auto                     combined   = device.create_buffer<int>(input_a_values.size());
        auto                     converted  = device.create_buffer<float>(unary_values.size());
        auto                     difference = device.create_buffer<int>(input_a_values.size());
        stream << unary.copy_from(luisa::span{unary_values})
               << input_a.copy_from(luisa::span{input_a_values})
               << input_b.copy_from(luisa::span{input_b_values});

        auto square       = [](const Int& value) noexcept { return value * value; };
        auto double_value = [](const Int& value) noexcept { return value * 2; };
        auto combine      = [](const Int& lhs, const Int& rhs) noexcept { return lhs + rhs * 2; };
        auto split        = [](const Int& lhs, const Int& rhs) noexcept
        { return std::make_tuple(lhs + rhs, lhs - rhs); };
        CommandList commands;
        transform.Transform(
            commands, BufferView<uint>{}, unary.view(), converted.view(), unary_values.size(), double_value);
        transform.Transform(commands, BufferView<uint>{}, unary.view(), unary.view(), unary_values.size(), square);
        transform.Transform(commands,
                            BufferView<uint>{},
                            std::make_tuple(input_a.view(), input_b.view()),
                            combined.view(),
                            input_a_values.size(),
                            combine);
        transform.Transform(commands,
                            BufferView<uint>{},
                            std::make_tuple(input_a.view(), input_b.view()),
                            std::make_tuple(input_a.view(), difference.view()),
                            input_a_values.size(),
                            split);

        luisa::vector<int>   unary_result(unary_values.size());
        luisa::vector<int>   input_a_result(input_a_values.size());
        luisa::vector<int>   combined_result(input_a_values.size());
        luisa::vector<float> converted_result(unary_values.size());
        luisa::vector<int>   difference_result(input_a_values.size());
        stream << commands.commit() << unary.copy_to(luisa::span{unary_result})
               << input_a.copy_to(luisa::span{input_a_result})
               << combined.copy_to(luisa::span{combined_result})
               << converted.copy_to(luisa::span{converted_result})
               << difference.copy_to(luisa::span{difference_result}) << synchronize();
        for(size_t index = 0u; index < unary_values.size(); ++index)
        {
            expect(unary_result[index] == unary_values[index] * unary_values[index]);
            expect(converted_result[index] == static_cast<float>(unary_values[index] * 2));
            expect(combined_result[index] == input_a_values[index] + input_b_values[index] * 2);
            expect(input_a_result[index] == input_a_values[index] + input_b_values[index]);
            expect(difference_result[index] == input_a_values[index] - input_b_values[index]);
        }
    };

    "device_transform_if_leaves_rejected_outputs_untouched"_test = [&]
    {
        constexpr int            sentinel = -999;
        const luisa::vector<int> input_a_values{1, 2, 3, 4};
        const luisa::vector<int> input_b_values{10, 20, 30, 40};
        const luisa::vector<int> initial(input_a_values.size(), sentinel);
        auto                     input_a = device.create_buffer<int>(input_a_values.size());
        auto                     input_b = device.create_buffer<int>(input_b_values.size());
        auto                     output  = device.create_buffer<int>(input_a_values.size());
        stream << input_a.copy_from(luisa::span{input_a_values})
               << input_b.copy_from(luisa::span{input_b_values}) << output.copy_from(luisa::span{initial});

        auto        predicate = [](const Int& lhs, const Int&) noexcept { return lhs % 2 == 0; };
        auto        add       = [](const Int& lhs, const Int& rhs) noexcept { return lhs + rhs; };
        CommandList commands;
        transform.TransformIf(commands,
                              BufferView<uint>{},
                              std::make_tuple(input_a.view(), input_b.view()),
                              output.view(),
                              input_a_values.size(),
                              predicate,
                              add);

        luisa::vector<int> result(input_a_values.size());
        stream << commands.commit() << output.copy_to(luisa::span{result}) << synchronize();
        for(size_t index = 0u; index < input_a_values.size(); ++index)
        {
            int expected =
                input_a_values[index] % 2 == 0 ? input_a_values[index] + input_b_values[index] : sentinel;
            expect(result[index] == expected);
        }
    };

    "device_transform_stable_address_entry_points_preserve_values"_test = [&]
    {
        const luisa::vector<int> input_values{2, 3, 4, 5};
        const luisa::vector<int> other_values{10, 20, 30, 40};
        auto                     input        = device.create_buffer<int>(input_values.size());
        auto                     other        = device.create_buffer<int>(other_values.size());
        auto                     unary_output = device.create_buffer<int>(input_values.size());
        auto                     tuple_output = device.create_buffer<int>(input_values.size());
        stream << input.copy_from(luisa::span{input_values}) << other.copy_from(luisa::span{other_values});

        auto        increment = [](const Int& value) noexcept { return value + 1; };
        auto        multiply  = [](const Int& lhs, const Int& rhs) noexcept { return lhs * rhs; };
        CommandList commands;
        transform.TransformStableArgumentAddresses(
            commands, BufferView<uint>{}, input.view(), unary_output.view(), input_values.size(), increment);
        transform.TransformStableArgumentAddresses(commands,
                                                   BufferView<uint>{},
                                                   std::make_tuple(input.view(), other.view()),
                                                   tuple_output.view(),
                                                   input_values.size(),
                                                   multiply);

        luisa::vector<int> unary_result(input_values.size());
        luisa::vector<int> tuple_result(input_values.size());
        stream << commands.commit() << unary_output.copy_to(luisa::span{unary_result})
               << tuple_output.copy_to(luisa::span{tuple_result}) << synchronize();
        for(size_t index = 0u; index < input_values.size(); ++index)
        {
            expect(unary_result[index] == input_values[index] + 1);
            expect(tuple_result[index] == input_values[index] * other_values[index]);
        }
    };
}
