use dlpark::traits::IntoDlpack;

fn main() {
    let arr = Box::new((vec![1.0f32, 2., 3.], vec![3i64]));
    let pack = arr.into_dlpack();
}
